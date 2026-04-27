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
        """v1: basic profile. v2: real video analysis (duration, motion, occlusion)."""
        return VideoProfile(
            duration_s=0.0,
            camera_motion_score=0.0,
            occlusion_score=0.0,
            has_multiple_shots=False,
            requires_gait_analysis=False,
        )

    def process_video_path(
        self,
        path: str,
        person_height_cm: float = 170.0,
    ) -> SkeletonSequence:
        return self.process(VideoInput(path=path, person_height_cm=person_height_cm))
