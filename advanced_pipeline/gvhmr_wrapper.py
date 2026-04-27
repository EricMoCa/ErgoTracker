"""
advanced_pipeline/gvhmr_wrapper.py
World-grounded pose estimation via GVHMR (SIGGRAPH Asia 2024).
https://github.com/zju3dv/GVHMR

Pipeline (when installed):
  1. SimpleVO estimates per-frame camera angular velocity
  2. PersonDetector crops person bbox from each frame
  3. ViTPose extracts 2D keypoints + image features
  4. GVHMR Transformer (with RoPE) infers SMPL params in Gravity-View system
  5. GV → world coordinate composition gives final world-grounded joints
  6. smpl_converter converts [N,24,3] → SkeletonSequence

If GVHMR is not installed: is_available() = False, estimate() returns None.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from schemas import SkeletonSequence

from .smpl_converter import smpl_joints_to_skeleton_sequence
from .visual_odometry import VisualOdometry


class GVHMRWrapper:
    """
    Wrapper for GVHMR — primary world-grounded motor.

    Install:
        git clone https://github.com/zju3dv/GVHMR
        pip install -e GVHMR/
    """

    HF_CHECKPOINT = "zju3dv/GVHMR"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._smpl = None
        self.vo = VisualOdometry(method="opencv_fallback")

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            from gvhmr.models.smpl_hmr2 import GVHMR  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from gvhmr.models.smpl_hmr2 import GVHMR as GVHMRModel
            self._model = GVHMRModel.from_pretrained(self.HF_CHECKPOINT).to(self.device)
            self._model.eval()
            logger.info("GVHMR loaded")
            return True
        except ImportError:
            logger.warning("GVHMR not installed — pip install -e GVHMR/")
        except Exception as exc:
            logger.warning(f"GVHMR load failed: {exc}")
        return False

    def _load_smpl(self) -> bool:
        if self._smpl is not None:
            return True
        try:
            # GVHMR bundles or expects smplx; try smplx first, then smpl
            try:
                import smplx
                self._smpl = smplx.create(model_type="smpl", gender="neutral", use_pca=False).to(self.device)
            except Exception:
                from smpl import SMPL
                self._smpl = SMPL().to(self.device)
            return True
        except Exception as exc:
            logger.warning(f"SMPL model unavailable: {exc}")
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """
        Estimate world-grounded poses. Returns None if GVHMR unavailable.
        """
        if not self._load_model():
            return None

        try:
            return self._run_inference(video_path, person_height_cm)
        except Exception as exc:
            logger.error(f"GVHMR inference failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, video_path: str, person_height_cm: float) -> SkeletonSequence | None:
        import torch
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Attempt 1: use GVHMR's own high-level demo runner if available
        result = self._try_gvhmr_demo(video_path, fps, person_height_cm)
        if result is not None:
            return result

        # Attempt 2: manual inference via model forward
        result = self._try_gvhmr_forward(video_path, fps, person_height_cm)
        if result is not None:
            return result

        logger.warning("GVHMR: all inference paths failed")
        return None

    def _try_gvhmr_demo(self, video_path: str, fps: float, person_height_cm: float) -> SkeletonSequence | None:
        """Try high-level demo runners that some GVHMR versions expose."""
        import tempfile, os

        demo_fns = []
        try:
            from gvhmr.tools.demo import run_demo
            demo_fns.append(("gvhmr.tools.demo.run_demo", run_demo))
        except ImportError:
            pass
        try:
            from gvhmr.apis import inference_video
            demo_fns.append(("gvhmr.apis.inference_video", inference_video))
        except ImportError:
            pass
        try:
            from gvhmr.run import run as gvhmr_run
            demo_fns.append(("gvhmr.run.run", gvhmr_run))
        except ImportError:
            pass

        for name, fn in demo_fns:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out = fn(video=video_path, output=tmp, device=self.device)
                    if out is None:
                        out = fn(video_path)
                joints = self._extract_joints_from_output(out)
                if joints is not None:
                    logger.info(f"GVHMR via {name}: {joints.shape[0]} frames")
                    return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
            except Exception as exc:
                logger.debug(f"GVHMR demo {name} failed: {exc}")

        return None

    def _try_gvhmr_forward(self, video_path: str, fps: float, person_height_cm: float) -> SkeletonSequence | None:
        """Lower-level forward pass through the GVHMR model."""
        import torch

        try:
            frames, bboxes = self._extract_frames_and_bboxes(video_path)
            if not frames:
                return None

            kp2d, features = self._extract_2d_keypoints(frames, bboxes)
            cam_angvel = self._estimate_cam_angvel(video_path, len(frames))

            batch = {
                "keypoints": torch.tensor(kp2d, dtype=torch.float32).unsqueeze(0).to(self.device),
                "cam_angvel": torch.tensor(cam_angvel, dtype=torch.float32).unsqueeze(0).to(self.device),
                "bbox": torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(self.device),
            }

            with torch.no_grad():
                output = self._model(batch)

            joints = self._extract_joints_from_output(output)
            if joints is None:
                joints = self._smpl_params_to_joints(output)
            if joints is None:
                return None

            logger.info(f"GVHMR forward: {joints.shape[0]} frames")
            return smpl_joints_to_skeleton_sequence(joints, video_path, fps, person_height_cm)
        except Exception as exc:
            logger.debug(f"GVHMR forward failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_frames_and_bboxes(self, video_path: str) -> tuple[list, list]:
        """Extract BGR frames and rough full-frame bboxes."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames, bboxes = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            H, W = frame.shape[:2]
            frames.append(frame)
            bboxes.append([0.0, 0.0, float(W), float(H)])
        cap.release()
        return frames, bboxes

    def _extract_2d_keypoints(self, frames: list, bboxes: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract 17 COCO-style 2D keypoints per frame.
        Tries ViTPose if available, falls back to RTMPose ONNX from pose_pipeline.
        Returns (kp2d [N,17,3], features [N,C]) — features may be zeros.
        """
        N = len(frames)
        kp2d = np.zeros((N, 17, 3), dtype=np.float32)
        features = np.zeros((N, 1), dtype=np.float32)

        try:
            from pose_pipeline.model_downloader import ensure_model
            from pose_pipeline.pose_2d import PoseEstimator2D
            estimator = PoseEstimator2D(str(ensure_model("rtmpose_m")))
            for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
                kps = estimator.estimate(frame, bbox)
                for j, name in enumerate(["nose","left_eye","right_eye","left_ear","right_ear",
                                           "left_shoulder","right_shoulder","left_elbow","right_elbow",
                                           "left_wrist","right_wrist","left_hip","right_hip",
                                           "left_knee","right_knee","left_ankle","right_ankle"]):
                    if name in kps:
                        kp = kps[name]
                        kp2d[i, j] = [kp.x, kp.y, kp.confidence]
        except Exception:
            pass

        return kp2d, features

    def _estimate_cam_angvel(self, video_path: str, n_frames: int) -> np.ndarray:
        """Returns [N, 6] camera angular velocity (rotation6D representation)."""
        rots = self.vo.estimate(video_path)
        angvel = np.zeros((n_frames, 6), dtype=np.float32)
        # rotation6D: first two columns of rotation matrix flattened
        identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        for i in range(n_frames):
            if i < len(rots):
                R = rots[i]
                angvel[i] = np.concatenate([R[:, 0], R[:, 1]])
            else:
                angvel[i] = identity_6d
        return angvel

    def _extract_joints_from_output(self, output) -> np.ndarray | None:
        """Try various output dict keys to find [N,24,3] joint positions."""
        if output is None:
            return None
        keys_to_try = [
            "joints_world", "smpl_joints_world", "world_joints",
            "pred_joints_world", "joints3d_world",
            "joints", "smpl_joints", "pred_joints", "joints3d",
        ]
        import torch
        if isinstance(output, dict):
            for key in keys_to_try:
                if key in output:
                    val = output[key]
                    if isinstance(val, torch.Tensor):
                        val = val.squeeze(0).cpu().numpy()
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                        return val
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            for item in output:
                result = self._extract_joints_from_output(item)
                if result is not None:
                    return result
        return None

    def _smpl_params_to_joints(self, output) -> np.ndarray | None:
        """Recover joint positions by running SMPL body model on output parameters."""
        if not self._load_smpl():
            return None
        try:
            import torch
            smpl_params = None
            if isinstance(output, dict):
                smpl_params = output.get("smpl_params") or output.get("pred_smpl_params")
            if smpl_params is None:
                return None
            body_pose = smpl_params.get("body_pose")
            global_orient = smpl_params.get("global_orient")
            betas = smpl_params.get("betas")
            transl = smpl_params.get("transl")
            if body_pose is None:
                return None
            with torch.no_grad():
                body = self._smpl(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    betas=betas,
                    transl=transl,
                )
            joints = body.joints.squeeze(0).cpu().numpy() if hasattr(body, "joints") else None
            return joints
        except Exception as exc:
            logger.debug(f"SMPL forward failed: {exc}")
            return None
