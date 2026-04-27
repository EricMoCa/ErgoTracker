"""
advanced_pipeline/smpl_converter.py
Converts SMPL/SMPL-X joint positions (numpy [N, J, 3]) to SkeletonSequence.

All world-grounded motors (GVHMR, WHAM, TRAM, HumanMM) output SMPL joint
positions in metres. This module provides the single conversion path.

SMPL-24 joint index → ErgoTracker keypoint name mapping is fixed and does not
depend on any external package.
"""
from __future__ import annotations

import numpy as np

from schemas import Keypoint3D, Skeleton3D, SkeletonSequence, KEYPOINT_NAMES

# ---------------------------------------------------------------------------
# SMPL-24 joint indices → keypoint name
# ---------------------------------------------------------------------------
# Standard SMPL joint ordering (24 joints):
#   0  pelvis        → mid_hip
#   1  left_hip      → left_hip
#   2  right_hip     → right_hip
#   3  spine1        (skip — torso internal)
#   4  left_knee     → left_knee
#   5  right_knee    → right_knee
#   6  spine2        (skip)
#   7  left_ankle    → left_ankle
#   8  right_ankle   → right_ankle
#   9  spine3        (skip)
#   10 left_foot     (skip — not in KEYPOINT_NAMES)
#   11 right_foot    (skip)
#   12 neck          → neck
#   13 left_collar   (skip)
#   14 right_collar  (skip)
#   15 head          → nose  (closest approximation)
#   16 left_shoulder → left_shoulder
#   17 right_shoulder→ right_shoulder
#   18 left_elbow    → left_elbow
#   19 right_elbow   → right_elbow
#   20 left_wrist    → left_wrist
#   21 right_wrist   → right_wrist
#   22 left_hand     (skip)
#   23 right_hand    (skip)

_SMPL24_TO_KP: dict[int, str] = {
    0:  "mid_hip",
    1:  "left_hip",
    2:  "right_hip",
    4:  "left_knee",
    5:  "right_knee",
    7:  "left_ankle",
    8:  "right_ankle",
    12: "neck",
    15: "nose",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}

# SMPL-45 wholebody: first 24 joints are identical to SMPL-24.
# Joint 25+ are face / hands — not needed for ergonomics.
_SMPL45_TO_KP = _SMPL24_TO_KP  # same mapping; extra joints are ignored

# Minimal confidence assigned to all SMPL-derived joints
_SMPL_CONFIDENCE = 0.9


def smpl_joints_to_skeleton_sequence(
    joints_world: np.ndarray,
    video_path: str,
    fps: float,
    person_height_cm: float,
    frame_indices: list[int] | None = None,
) -> SkeletonSequence:
    """
    Convert SMPL joint positions to SkeletonSequence.

    Args:
        joints_world: float32 array of shape [N, J, 3] in metres,
                      J = 24 (SMPL) or 45 (SMPL-X wholebody).
                      Y-axis = up, origin = arbitrary world origin.
        video_path:   source video path (stored in SkeletonSequence metadata).
        fps:          frame-rate of the source video.
        person_height_cm: used to compute scale_px_to_m.
        frame_indices: original frame indices if joints_world is a sub-sample.
    """
    N, J, _ = joints_world.shape
    mapping = _SMPL24_TO_KP if J <= 24 else _SMPL45_TO_KP

    if frame_indices is None:
        frame_indices = list(range(N))

    skeletons: list[Skeleton3D] = []
    for i, frame_idx in enumerate(frame_indices):
        kps_xyz = joints_world[i]  # [J, 3]
        keypoints = _build_keypoints(kps_xyz, mapping)
        skel = Skeleton3D(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / fps,
            keypoints=keypoints,
            scale_px_to_m=1.0,
            person_height_cm=person_height_cm,
            coordinate_system="world",
        )
        skeletons.append(skel)

    return SkeletonSequence(
        video_path=video_path,
        fps=fps,
        total_frames=frame_indices[-1] + 1 if frame_indices else N,
        frames=skeletons,
    )


def _build_keypoints(
    kps_xyz: np.ndarray,
    mapping: dict[int, str],
) -> dict[str, Keypoint3D]:
    """
    Build a keypoints dict from SMPL joint positions for a single frame.
    Derives mid_shoulder, right_eye, left_eye from available joints.
    """
    kp: dict[str, Keypoint3D] = {}

    for smpl_idx, name in mapping.items():
        if smpl_idx >= len(kps_xyz):
            continue
        x, y, z = float(kps_xyz[smpl_idx, 0]), float(kps_xyz[smpl_idx, 1]), float(kps_xyz[smpl_idx, 2])
        kp[name] = Keypoint3D(x=x, y=y, z=z, confidence=_SMPL_CONFIDENCE)

    # Derived keypoints
    _add_derived(kp, kps_xyz)

    # Fill any remaining KEYPOINT_NAMES not covered with low-confidence zeros
    for name in KEYPOINT_NAMES:
        if name not in kp:
            # Use neck as approximate fallback for unmapped names
            ref = kp.get("neck") or kp.get("mid_hip")
            if ref:
                kp[name] = Keypoint3D(x=ref.x, y=ref.y, z=ref.z, confidence=0.1, occluded=True)
            else:
                kp[name] = Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0, occluded=True)

    return kp


def _add_derived(kp: dict[str, Keypoint3D], kps_xyz: np.ndarray) -> None:
    """Compute mid_shoulder, right_eye, left_eye from available joints."""
    ls = kp.get("left_shoulder")
    rs = kp.get("right_shoulder")
    if ls and rs:
        kp["mid_shoulder"] = Keypoint3D(
            x=(ls.x + rs.x) / 2,
            y=(ls.y + rs.y) / 2,
            z=(ls.z + rs.z) / 2,
            confidence=min(ls.confidence, rs.confidence),
        )

    nose = kp.get("nose")
    neck = kp.get("neck")
    if nose and neck:
        # Eyes are slightly lateral and forward relative to nose
        dx = (nose.x - neck.x) * 0.1
        dz = (nose.z - neck.z) * 0.1
        kp["right_eye"] = Keypoint3D(x=nose.x + dz, y=nose.y, z=nose.z - dx, confidence=0.7)
        kp["left_eye"]  = Keypoint3D(x=nose.x - dz, y=nose.y, z=nose.z + dx, confidence=0.7)
