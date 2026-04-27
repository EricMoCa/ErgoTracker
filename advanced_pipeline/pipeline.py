import numpy as np

from schemas import VideoInput, SkeletonSequence
from pose_pipeline import PosePipeline
from .pipeline_router import PipelineRouter, VideoProfile, HardwareProfile
from .gvhmr_wrapper import GVHMRWrapper
from .wham_wrapper import WHAMWrapper
from .tram_wrapper import TRAMWrapper
from .contact_refinement import ContactRefinement
from .stride_refinement import STRIDERefinement
from .humanmm_wrapper import HumanMMWrapper
from loguru import logger


class AdvancedPosePipeline:
    """
    Advanced pose pipeline — drop-in replacement for PosePipeline.
    PipelineRouter selects the best available world-grounded motor.
    Always returns a valid SkeletonSequence (falls back to CPU PosePipeline).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._verify_gpu()
        self._router = PipelineRouter()
        self._gvhmr = GVHMRWrapper(self.device)
        self._wham = WHAMWrapper(self.device)
        self._tram = TRAMWrapper(self.device)
        self._stride = STRIDERefinement()
        self._humanmm = HumanMMWrapper(self.device)
        self._contact = ContactRefinement()
        self._fallback = PosePipeline(device="cpu")

    def _verify_gpu(self) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("GPU not available. Using CPU.")
                self.device = "cpu"
                return
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_vram < 2.5:
                logger.warning(
                    f"Free VRAM: {free_vram:.1f} GB — verify that Ollama released GPU."
                )
        except Exception:
            self.device = "cpu"

    def process(self, video_input: VideoInput) -> SkeletonSequence:
        hw = HardwareProfile.detect()
        video_profile = self._build_video_profile(video_input)

        motor = self._router.select(
            video_profile=video_profile,
            hardware=hw,
            gvhmr_available=self._gvhmr.is_available(),
            wham_available=self._wham.is_available(),
            tram_available=self._tram.is_available(),
            humanmm_available=self._humanmm.is_available(),
        )

        logger.info(f"AdvancedPosePipeline → motor: {motor}")
        result: SkeletonSequence | None = None

        if motor == "humanmm":
            result = self._humanmm.process_multishot(
                video_input.path, video_input.person_height_cm
            )
        elif motor == "gvhmr_with_contact":
            result = self._gvhmr.estimate(video_input.path, video_input.person_height_cm)
            if result:
                result = self._contact.refine(result)
        elif motor == "gvhmr":
            result = self._gvhmr.estimate(video_input.path, video_input.person_height_cm)
        elif motor == "wham":
            seq, _ = self._wham.estimate(video_input.path, video_input.person_height_cm)
            result = seq
        elif motor == "tram":
            result = self._tram.estimate(video_input.path, video_input.person_height_cm)

        if result and video_profile.occlusion_score > 0.2:
            result = self._stride.refine(result)

        if result:
            return result

        logger.warning("Falling back to base PosePipeline (CPU).")
        return self._fallback.process(video_input)

    def _build_video_profile(self, video_input: VideoInput) -> VideoProfile:
        """
        Analyse the video with OpenCV to build a real VideoProfile:
        - duration from frame count / fps
        - camera_motion_score from mean optical-flow magnitude between sampled frames
        - has_multiple_shots from histogram correlation drops
        - occlusion_score estimated from how many sampled frames lack a clear person bbox
        """
        try:
            import cv2
        except ImportError:
            return VideoProfile(duration_s=0.0)

        cap = cv2.VideoCapture(video_input.path)
        if not cap.isOpened():
            return VideoProfile(duration_s=0.0)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total / fps if fps > 0 else 0.0

        # Sample at most 60 frames evenly for speed
        sample_step = max(1, total // 60)
        sampled: list[np.ndarray] = []
        for fi in range(0, total, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(sampled) < 2:
            return VideoProfile(duration_s=duration_s)

        # Camera motion: mean optical flow magnitude between consecutive sampled frames
        motion_scores: list[float] = []
        shot_boundaries = 0
        prev_hist = None
        flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        for i in range(1, len(sampled)):
            flow = cv2.calcOpticalFlowFarneback(sampled[i - 1], sampled[i], None, **flow_params)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(float(mag.mean()))

            # Shot detection via histogram correlation
            h_curr = cv2.calcHist([sampled[i]], [0], None, [64], [0, 256])
            cv2.normalize(h_curr, h_curr)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, h_curr, cv2.HISTCMP_CORREL)
                if corr < 0.45:
                    shot_boundaries += 1
            prev_hist = h_curr

        # Normalise motion: clamp to [0,1] using empirical scale (5 px/frame ≈ 1.0)
        raw_motion = float(np.mean(motion_scores)) if motion_scores else 0.0
        camera_motion_score = min(1.0, raw_motion / 5.0)

        # Occlusion estimation: frames where optical flow is very low everywhere
        # (person not visible → detector would fail → pose unreliable)
        low_motion_frames = sum(1 for s in motion_scores if s < 0.3)
        occlusion_score = min(1.0, low_motion_frames / max(len(motion_scores), 1))

        logger.debug(
            f"VideoProfile: duration={duration_s:.1f}s  motion={camera_motion_score:.2f}"
            f"  shots={shot_boundaries}  occlusion={occlusion_score:.2f}"
        )

        return VideoProfile(
            duration_s=duration_s,
            has_multiple_shots=shot_boundaries >= 2,
            camera_motion_score=camera_motion_score,
            occlusion_score=occlusion_score,
            requires_gait_analysis=False,
        )

    def process_video_path(
        self,
        path: str,
        person_height_cm: float = 170.0,
    ) -> SkeletonSequence:
        return self.process(VideoInput(path=path, person_height_cm=person_height_cm))
