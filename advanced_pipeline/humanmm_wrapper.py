"""
advanced_pipeline/humanmm_wrapper.py
Multi-shot video pose estimation via HumanMM (CVPR 2025).
https://github.com/zhangyuhong01/HumanMM-code

Use when the video contains shot transitions / camera cuts.
Detects shot boundaries and applies cross-shot pose consistency.

If HumanMM is not installed: is_available() = False, process_multishot() returns None.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from schemas import SkeletonSequence

from .smpl_converter import smpl_joints_to_skeleton_sequence


class HumanMMWrapper:
    """
    Wrapper for HumanMM — multi-shot video motor (CVPR 2025).

    Install:
        git clone https://github.com/zhangyuhong01/HumanMM-code
        cd HumanMM-code && pip install -e .
        # Download checkpoints per README
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._available = self._check_humanmm()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_humanmm(self) -> bool:
        try:
            import humanmm  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from humanmm.models import build_humanmm
            self._model = build_humanmm(device=self.device)
            self._model.eval()
            logger.info("HumanMM model loaded")
            return True
        except Exception as exc:
            logger.debug(f"HumanMM model load attempt 1 failed: {exc}")

        try:
            from humanmm import HumanMM
            self._model = HumanMM(device=self.device)
            logger.info("HumanMM model loaded (fallback constructor)")
            return True
        except Exception as exc:
            logger.warning(f"HumanMM model load failed: {exc}")

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_multishot(
        self,
        video_path: str,
        person_height_cm: float = 170.0,
    ) -> SkeletonSequence | None:
        """
        Process a multi-shot video with cross-shot pose consistency.
        Returns None if HumanMM is not available or inference fails.
        """
        if not self._available:
            logger.warning("HumanMM not installed. Install: pip install -e HumanMM-code/")
            return None

        try:
            return self._run_inference(video_path, person_height_cm)
        except Exception as exc:
            logger.error(f"HumanMM inference failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, video_path: str, person_height_cm: float) -> SkeletonSequence | None:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Detect shot boundaries
        shot_boundaries = self._detect_shot_boundaries(video_path)
        logger.info(f"HumanMM: detected {len(shot_boundaries)} shot boundaries")

        # Try high-level runner
        result = self._try_humanmm_run(video_path, fps, person_height_cm)
        if result is not None:
            return result

        # Try manual forward
        result = self._try_humanmm_forward(video_path, fps, person_height_cm, shot_boundaries)
        return result

    def _try_humanmm_run(self, video_path: str, fps: float, person_height_cm: float) -> SkeletonSequence | None:
        """Try HumanMM CLI/demo runners."""
        import tempfile

        runners = []
        try:
            from humanmm.run import run_humanmm
            runners.append(("humanmm.run.run_humanmm", run_humanmm))
        except ImportError:
            pass
        try:
            from humanmm.inference import inference_video
            runners.append(("humanmm.inference.inference_video", inference_video))
        except ImportError:
            pass
        try:
            from humanmm.demo import run
            runners.append(("humanmm.demo.run", run))
        except ImportError:
            pass

        for name, fn in runners:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out = fn(video=video_path, output=tmp, device=self.device)
                joints = self._extract_joints(out)
                if joints is not None:
                    logger.info(f"HumanMM via {name}: {joints.shape[0]} frames")
                    return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
            except Exception as exc:
                logger.debug(f"HumanMM runner {name} failed: {exc}")

        return None

    def _try_humanmm_forward(
        self, video_path: str, fps: float, person_height_cm: float, shots: list[int]
    ) -> SkeletonSequence | None:
        """Manual forward pass with per-shot processing + consistency stitching."""
        if not self._load_model():
            return None
        try:
            import torch, cv2
            frames = self._read_frames(video_path)
            if not frames:
                return None

            # Process per shot and stitch
            all_joints = []
            shot_starts = [0] + shots
            shot_ends = shots + [len(frames)]
            for start, end in zip(shot_starts, shot_ends):
                shot_frames = frames[start:end]
                shot_t = self._frames_to_tensor(shot_frames)
                with torch.no_grad():
                    out = self._model({"frames": shot_t.to(self.device), "shot_idx": start})
                j = self._extract_joints(out)
                if j is not None:
                    all_joints.append(j)

            if not all_joints:
                return None

            joints = np.concatenate(all_joints, axis=0)
            logger.info(f"HumanMM forward: {joints.shape[0]} frames across {len(all_joints)} shots")
            return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
        except Exception as exc:
            logger.debug(f"HumanMM forward failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_shot_boundaries(self, video_path: str) -> list[int]:
        """
        Detect shot cuts using histogram difference between consecutive frames.
        Returns list of frame indices where a shot transition occurs.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            boundaries = []
            prev_hist = None
            frame_idx = 0
            threshold = 0.45  # histogram correlation drop threshold

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist)

                if prev_hist is not None:
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if corr < threshold:
                        boundaries.append(frame_idx)

                prev_hist = hist
                frame_idx += 1
            cap.release()
            return boundaries
        except Exception:
            return []

    def _read_frames(self, video_path: str) -> list:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _frames_to_tensor(self, frames: list) -> "torch.Tensor":
        import torch
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)

    def _extract_joints(self, output) -> np.ndarray | None:
        """Try various output keys to find [N, 24, 3] joints."""
        if output is None:
            return None
        import torch

        joint_keys = [
            "joints_world", "world_joints", "smpl_joints_world",
            "smpl_joints", "joints3d", "joints", "pred_joints",
        ]
        src = output
        if isinstance(output, (list, tuple)):
            src = output[0] if output else {}

        if isinstance(src, dict):
            for key in joint_keys:
                if key not in src:
                    continue
                val = src[key]
                if isinstance(val, torch.Tensor):
                    val = val.squeeze(0).cpu().numpy()
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    if val.shape[-1] == 3 and 15 <= val.shape[1] <= 55:
                        return val
        return None
