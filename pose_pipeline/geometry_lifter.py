"""
pose_pipeline/geometry_lifter.py
Geometric 2D→3D lifting — fallback when MotionBERT ONNX is unavailable.

Uses anthropometric proportions and pixel-space geometry to estimate 3D
keypoint positions from RTMPose 2D detections.  No neural network required.

Accuracy: sufficient for frontal/lateral industrial footage (~±10°  on trunk
and shoulder angles).  Not suitable for complex occluded poses.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from schemas import KEYPOINT_NAMES
from schemas.skeleton import Keypoint3D

from .pose_2d import Keypoint2D

# ---------------------------------------------------------------------------
# Relative depth offsets in metres (Z = forward from camera).
# For a person standing upright facing the camera these are approximately 0.
# Shoulders / elbows / wrists can be in front of or behind the torso plane.
# We use a heuristic: joints with large arm-torso angle get a small Z offset.
# ---------------------------------------------------------------------------
_BASE_Z: dict[str, float] = {
    "nose":           0.05,
    "left_eye":       0.05,
    "right_eye":      0.05,
    "neck":           0.0,
    "mid_shoulder":   0.0,
    "left_shoulder":  0.0,
    "right_shoulder": 0.0,
    "left_elbow":     0.0,   # updated dynamically from arm angle
    "right_elbow":    0.0,
    "left_wrist":     0.0,
    "right_wrist":    0.0,
    "mid_hip":        0.0,
    "left_hip":       0.0,
    "right_hip":      0.0,
    "left_knee":      0.0,
    "right_knee":     0.0,
    "left_ankle":     0.0,
    "right_ankle":    0.0,
}


class GeometricLifter:
    """
    Drop-in replacement for PoseLifter3D when MotionBERT ONNX is unavailable.

    Input:  list[dict[str, Keypoint2D]]  (pixel coordinates from RTMPose)
    Output: list[dict[str, Keypoint3D]]  (metric coordinates, camera-relative)
    """

    def lift(
        self,
        keypoints_2d_sequence: list[dict[str, Keypoint2D]],
        person_height_cm: float = 170.0,
        img_w: float = 1920.0,
        img_h: float = 1080.0,
    ) -> list[dict[str, Keypoint3D]]:
        """
        Lift a sequence of 2D keypoints to 3D using geometric heuristics.
        """
        if not keypoints_2d_sequence:
            return []

        result: list[dict[str, Keypoint3D]] = []
        for kps_2d in keypoints_2d_sequence:
            kps_3d = self._lift_frame(kps_2d, person_height_cm, img_w, img_h)
            result.append(kps_3d)

        logger.debug(f"GeometricLifter: lifted {len(result)} frames")
        return result

    def _lift_frame(
        self,
        kps_2d: dict[str, Keypoint2D],
        person_height_cm: float,
        img_w: float,
        img_h: float,
    ) -> dict[str, Keypoint3D]:
        """Lift one frame."""
        h_m = person_height_cm / 100.0

        # Estimate pixel height of person from nose to ankles
        nose = kps_2d.get("nose")
        l_ankle = kps_2d.get("left_ankle")
        r_ankle = kps_2d.get("right_ankle")

        ankle_y = None
        if l_ankle and r_ankle and l_ankle.confidence > 0.3 and r_ankle.confidence > 0.3:
            ankle_y = (l_ankle.y + r_ankle.y) / 2.0
        elif l_ankle and l_ankle.confidence > 0.3:
            ankle_y = l_ankle.y
        elif r_ankle and r_ankle.confidence > 0.3:
            ankle_y = r_ankle.y

        pixel_height = None
        if nose and nose.confidence > 0.3 and ankle_y is not None:
            pixel_height = abs(ankle_y - nose.y) * 1.08  # 8% extra for head top

        if pixel_height is None or pixel_height < 20:
            # Fall back to image-height estimation: assume person ≈ 80% of image height
            pixel_height = img_h * 0.80

        scale_m_per_px = h_m / pixel_height

        # Reference Y: ankle position in pixels (floor = 0 m)
        ref_y_px = ankle_y if ankle_y is not None else img_h

        # Estimate torso depth from shoulder width
        l_sh = kps_2d.get("left_shoulder")
        r_sh = kps_2d.get("right_shoulder")
        z_scale = 0.0
        if l_sh and r_sh and l_sh.confidence > 0.3 and r_sh.confidence > 0.3:
            shoulder_w_px = abs(r_sh.x - l_sh.x)
            # Standard shoulder width ~45 cm; derive Z perturbation from asymmetry
            shoulder_w_m = shoulder_w_px * scale_m_per_px
            # If shoulders appear narrower than expected, person may be turned
            expected_shoulder_m = 0.42
            if shoulder_w_m < expected_shoulder_m:
                # Lateral turn → assign Z offset to shoulders
                turn_factor = 1.0 - (shoulder_w_m / expected_shoulder_m)
                z_scale = expected_shoulder_m * turn_factor * 0.5

        kps_3d: dict[str, Keypoint3D] = {}
        for name in KEYPOINT_NAMES:
            kp2 = kps_2d.get(name)
            if kp2 is None or kp2.confidence < 0.05:
                kps_3d[name] = Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0, occluded=True)
                continue

            # X: lateral position in metres from image centre
            x_m = (kp2.x - img_w / 2.0) * scale_m_per_px

            # Y: height in metres from floor
            y_m = (ref_y_px - kp2.y) * scale_m_per_px

            # Z: depth heuristic
            base_z = _BASE_Z.get(name, 0.0)
            z_m = base_z + self._dynamic_z(name, kps_2d, scale_m_per_px, z_scale)

            kps_3d[name] = Keypoint3D(
                x=float(x_m),
                y=float(y_m),
                z=float(z_m),
                confidence=float(kp2.confidence),
            )

        return kps_3d

    def _dynamic_z(
        self,
        name: str,
        kps_2d: dict[str, Keypoint2D],
        scale: float,
        turn_z: float,
    ) -> float:
        """Estimate Z from limb geometry."""
        side = "left" if name.startswith("left") else "right"
        sign = 1.0 if side == "left" else -1.0

        if name in ("left_shoulder", "right_shoulder", "mid_shoulder", "neck"):
            return sign * turn_z

        if name in ("left_elbow", "right_elbow"):
            sh = kps_2d.get(f"{side}_shoulder")
            el = kps_2d.get(f"{side}_elbow")
            if sh and el and sh.confidence > 0.3 and el.confidence > 0.3:
                # Elbow below shoulder → arm possibly forward/backward
                dy = (el.y - sh.y) * scale
                return dy * 0.3  # small Z from arm drop
            return sign * turn_z

        if name in ("left_wrist", "right_wrist"):
            el = kps_2d.get(f"{side}_elbow")
            wr = kps_2d.get(f"{side}_wrist")
            if el and wr and el.confidence > 0.3 and wr.confidence > 0.3:
                dx = (wr.x - el.x) * scale
                return -dx * 0.5  # wrist in front of elbow if arm forward
            return sign * turn_z * 1.2

        return 0.0
