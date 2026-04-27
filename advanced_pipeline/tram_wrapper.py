"""
advanced_pipeline/tram_wrapper.py
World-grounded estimation via TRAM (ECCV 2024).
https://github.com/yufu-wang/tram

Architecture:
  1. DROID-SLAM (with dual masking for dynamic objects) → metric camera trajectory
  2. VIMO (ViT-H video transformer) → kinematic body motion
  60% reduction in global trajectory error vs baselines.

If TRAM is not installed: is_available() = False, estimate() returns None.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from schemas import SkeletonSequence

from .smpl_converter import smpl_joints_to_skeleton_sequence


class TRAMWrapper:
    """
    Wrapper for TRAM — alternative world-grounded motor.

    Install:
        git clone https://github.com/yufu-wang/tram
        cd tram && pip install -e .
        # Also requires DROID-SLAM: see tram/README.md
        # Download TRAM checkpoints: bash fetch_models.sh
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._available = self._check_tram()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_tram(self) -> bool:
        try:
            import tram  # noqa: F401
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
            from tram.models import build_tram
            self._model = build_tram(device=self.device)
            self._model.eval()
            logger.info("TRAM model loaded")
            return True
        except Exception as exc:
            logger.debug(f"TRAM model load attempt 1 failed: {exc}")

        try:
            from tram import TRAM
            self._model = TRAM(device=self.device)
            logger.info("TRAM model loaded (fallback constructor)")
            return True
        except Exception as exc:
            logger.warning(f"TRAM model load failed: {exc}")

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """
        Estimate world-grounded poses via TRAM.
        Returns None if TRAM is not available or inference fails.
        """
        if not self._available:
            logger.warning("TRAM not installed. Install: pip install -e tram/")
            return None

        try:
            return self._run_inference(video_path, person_height_cm)
        except Exception as exc:
            logger.error(f"TRAM inference failed: {exc}")
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

        # Try TRAM high-level runner
        result = self._try_tram_run(video_path, fps, person_height_cm)
        if result is not None:
            return result

        # Try manual TRAM forward
        result = self._try_tram_forward(video_path, fps, person_height_cm)
        return result

    def _try_tram_run(self, video_path: str, fps: float, person_height_cm: float) -> SkeletonSequence | None:
        """Try TRAM's CLI/demo runner."""
        import tempfile

        runners = []
        try:
            from tram.run import run_tram
            runners.append(("tram.run.run_tram", run_tram))
        except ImportError:
            pass
        try:
            from tram.demo import run
            runners.append(("tram.demo.run", run))
        except ImportError:
            pass
        try:
            from tram.apis import inference_video
            runners.append(("tram.apis.inference_video", inference_video))
        except ImportError:
            pass

        for name, fn in runners:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out = fn(video=video_path, output=tmp, device=self.device)
                joints = self._extract_joints(out)
                if joints is not None:
                    logger.info(f"TRAM via {name}: {joints.shape[0]} frames")
                    return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
            except Exception as exc:
                logger.debug(f"TRAM runner {name} failed: {exc}")

        return None

    def _try_tram_forward(self, video_path: str, fps: float, person_height_cm: float) -> SkeletonSequence | None:
        """Manual TRAM model forward pass."""
        if not self._load_model():
            return None
        try:
            import torch
            video_data = self._preprocess_video(video_path)
            with torch.no_grad():
                output = self._model(video_data)
            joints = self._extract_joints(output)
            if joints is None:
                return None
            logger.info(f"TRAM forward: {joints.shape[0]} frames")
            return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
        except Exception as exc:
            logger.debug(f"TRAM forward failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_video(self, video_path: str) -> dict:
        """Basic video preprocessing — returns frame tensor."""
        import torch, cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [N,H,W,3]
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0)  # [1,N,3,H,W]
        return {"video": frames_t.to(self.device), "video_path": video_path}

    def _extract_joints(self, output) -> np.ndarray | None:
        """Try various output keys to find [N, 24, 3] joint array."""
        if output is None:
            return None
        import torch

        joint_keys = [
            "joints_world", "smpl_joints_world", "world_joints",
            "smpl_joints", "joints3d", "joints", "pred_joints",
            "verts",  # fall back to SMPL vertices (last resort — wrong shape)
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
