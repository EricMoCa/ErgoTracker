"""
advanced_pipeline/wham_wrapper.py
Foot-contact-aware pose estimation via WHAM (CVPR 2024).
https://github.com/yohanshin/WHAM

WHAM has two modes:
  - Full mode (Linux):   DPVO camera trajectory + RNN motion predictor
  - Local mode (Windows): skip DPVO, use identity camera — same body motion,
                           no global camera trajectory. Activated automatically.

If WHAM is not installed: is_available() = False, estimate() returns (None, None).
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from schemas import SkeletonSequence

from .smpl_converter import smpl_joints_to_skeleton_sequence


class WHAMWrapper:
    """
    Wrapper for WHAM — foot-ground contact analysis motor.

    Preferred alternative when gait analysis is needed:
      GVHMRWrapper + ContactRefinement  (more portable, no drift)

    Install:
        git clone https://github.com/yohanshin/WHAM
        cd WHAM
        pip install -r requirements.txt
        pip install -e third-party/ViTPose
        # Download SMPL from smpl.is.tue.mpg.de
        # Download WHAM checkpoint: bash fetch_demo_data.sh
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._cfg = None
        self._model = None
        self._detector = None
        self._available = self._check_wham()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_wham(self) -> bool:
        try:
            import wham  # noqa: F401
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
            from wham.configs import get_cfg_defaults
            from wham.models import build_network
            self._cfg = get_cfg_defaults()
            self._cfg.merge_from_file("configs/yamls/demo.yaml")
            self._model = build_network(self._cfg, self._cfg.TRAIN.CHECKPOINT)
            self._model.eval()
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            logger.info("WHAM model loaded")
            return True
        except Exception as exc:
            logger.warning(f"WHAM model load failed: {exc}")
        return False

    def _load_detector(self) -> bool:
        if self._detector is not None:
            return True
        try:
            from wham.utils.vitracked import Tracker
            self._detector = Tracker()
            return True
        except Exception:
            try:
                from wham.utils.tracks import get_tracks
                self._detector = get_tracks
                return True
            except Exception as exc:
                logger.debug(f"WHAM tracker load failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        video_path: str,
        person_height_cm: float = 170.0,
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        """
        Returns (SkeletonSequence, contact_probs) or (None, None) if unavailable.
        contact_probs: per-frame probability [0,1] that at least one foot is on ground.
        """
        if not self._available:
            logger.warning("WHAM not installed. See CLAUDE.md for installation.")
            return None, None

        try:
            return self._run_inference(video_path, person_height_cm)
        except Exception as exc:
            logger.error(f"WHAM inference failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self, video_path: str, person_height_cm: float
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Try high-level demo runner first
        result = self._try_wham_demo(video_path, fps, person_height_cm)
        if result[0] is not None:
            return result

        # Fall back to manual inference
        result = self._try_wham_forward(video_path, fps, person_height_cm)
        if result[0] is not None:
            return result

        logger.warning("WHAM: all inference paths failed")
        return None, None

    def _try_wham_demo(
        self, video_path: str, fps: float, person_height_cm: float
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        """Try WHAM's high-level demo / CLI runner."""
        import tempfile

        demo_fns = []
        try:
            from wham.run_demo import run_demo
            demo_fns.append(("wham.run_demo", run_demo))
        except ImportError:
            pass
        try:
            from wham.demo import run
            demo_fns.append(("wham.demo.run", run))
        except ImportError:
            pass

        for name, fn in demo_fns:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    # estimate_local_only=True skips DPVO (Windows-compatible)
                    out = fn(
                        video=video_path,
                        output=tmp,
                        estimate_local_only=True,  # Windows-safe
                        calib=None,
                    )
                joints, contact = self._parse_wham_output(out)
                if joints is not None:
                    logger.info(f"WHAM via {name}: {joints.shape[0]} frames (local mode)")
                    seq = smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
                    return seq, contact
            except Exception as exc:
                logger.debug(f"WHAM demo {name} failed: {exc}")

        return None, None

    def _try_wham_forward(
        self, video_path: str, fps: float, person_height_cm: float
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        """Manual WHAM forward pass."""
        if not self._load_model():
            return None, None
        try:
            import torch

            tracking_results = self._get_tracking_results(video_path)
            if not tracking_results:
                return None, None

            all_joints, all_contact = [], []
            for person_data in tracking_results[:1]:  # primary person only
                with torch.no_grad():
                    output = self._model(
                        person_data,
                        accumulate_token=True,
                        # Skip global trajectory for Windows compatibility
                    )
                joints, contact = self._parse_wham_output(output)
                if joints is not None:
                    all_joints.append(joints)
                    all_contact.append(contact or [0.0] * len(joints))

            if not all_joints:
                return None, None

            joints = all_joints[0]
            contact = all_contact[0]
            logger.info(f"WHAM forward: {joints.shape[0]} frames")
            seq = smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
            return seq, contact
        except Exception as exc:
            logger.debug(f"WHAM forward failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tracking_results(self, video_path: str) -> list:
        """Get per-person tracking results from WHAM's tracker."""
        if not self._load_detector():
            return []
        try:
            if callable(self._detector):
                return self._detector(video_path)
            return self._detector.track(video_path)
        except Exception as exc:
            logger.debug(f"WHAM tracking failed: {exc}")
            return []

    def _parse_wham_output(self, output) -> tuple[np.ndarray | None, list[float] | None]:
        """Extract joints [N,24,3] and contact_probs [N] from WHAM output dict."""
        if output is None:
            return None, None
        import torch

        joints = None
        contact = None

        joint_keys = ["joints_world", "joints3d_world", "smpl_joints", "joints", "pred_joints"]
        contact_keys = ["contact_probs", "contact", "foot_contact", "pred_contact"]

        src = output
        if isinstance(output, (list, tuple)):
            src = output[0] if output else {}

        if isinstance(src, dict):
            for key in joint_keys:
                if key in src:
                    val = src[key]
                    if isinstance(val, torch.Tensor):
                        val = val.squeeze(0).cpu().numpy()
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                        joints = val
                        break
            for key in contact_keys:
                if key in src:
                    val = src[key]
                    if isinstance(val, torch.Tensor):
                        val = val.squeeze().cpu().numpy()
                    if isinstance(val, np.ndarray):
                        # Reduce per-foot to single probability
                        contact = val.max(axis=-1).tolist() if val.ndim > 1 else val.tolist()
                        break

        return joints, contact
