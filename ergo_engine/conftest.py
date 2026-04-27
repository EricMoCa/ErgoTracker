import pytest
import numpy as np
from schemas import JointAngles, Skeleton3D, Keypoint3D, SkeletonSequence, KEYPOINT_NAMES


def make_angles(**kwargs) -> JointAngles:
    defaults = dict(frame_idx=0, timestamp_s=0.0)
    defaults.update(kwargs)
    return JointAngles(**defaults)


def make_upright_angles() -> JointAngles:
    """Postura completamente erguida."""
    return make_angles(
        trunk_flexion=0.0,
        trunk_lateral_bending=0.0,
        trunk_rotation=0.0,
        neck_flexion=0.0,
        neck_lateral_bending=0.0,
        shoulder_elevation_left=10.0,
        shoulder_elevation_right=10.0,
        elbow_flexion_left=80.0,
        elbow_flexion_right=80.0,
        wrist_flexion_left=0.0,
        wrist_flexion_right=0.0,
        wrist_deviation_left=0.0,
        wrist_deviation_right=0.0,
        knee_flexion_left=0.0,
        knee_flexion_right=0.0,
    )


def make_high_risk_angles() -> JointAngles:
    """Postura de alto riesgo: tronco >60°, cuello >20°, brazos elevados."""
    return make_angles(
        trunk_flexion=70.0,
        trunk_lateral_bending=20.0,
        neck_flexion=30.0,
        shoulder_elevation_left=100.0,
        shoulder_elevation_right=100.0,
        elbow_flexion_left=120.0,
        elbow_flexion_right=120.0,
        wrist_flexion_left=20.0,
        wrist_flexion_right=20.0,
        wrist_deviation_left=15.0,
        wrist_deviation_right=15.0,
        knee_flexion_left=70.0,
        knee_flexion_right=70.0,
    )


def make_skeleton(frame_idx: int = 0, upright: bool = True) -> Skeleton3D:
    """Skeleton sintético con keypoints en posición erecta o inclinada."""
    if upright:
        # Postura erecta: persona de 1.7m parada
        kps = {
            "nose":           Keypoint3D(x=0.0, y=1.7, z=0.0, confidence=0.9),
            "neck":           Keypoint3D(x=0.0, y=1.5, z=0.0, confidence=0.9),
            "right_shoulder": Keypoint3D(x=0.2, y=1.4, z=0.0, confidence=0.9),
            "right_elbow":    Keypoint3D(x=0.2, y=1.1, z=0.0, confidence=0.9),
            "right_wrist":    Keypoint3D(x=0.2, y=0.8, z=0.0, confidence=0.9),
            "left_shoulder":  Keypoint3D(x=-0.2, y=1.4, z=0.0, confidence=0.9),
            "left_elbow":     Keypoint3D(x=-0.2, y=1.1, z=0.0, confidence=0.9),
            "left_wrist":     Keypoint3D(x=-0.2, y=0.8, z=0.0, confidence=0.9),
            "right_hip":      Keypoint3D(x=0.1, y=0.9, z=0.0, confidence=0.9),
            "right_knee":     Keypoint3D(x=0.1, y=0.5, z=0.0, confidence=0.9),
            "right_ankle":    Keypoint3D(x=0.1, y=0.1, z=0.0, confidence=0.9),
            "left_hip":       Keypoint3D(x=-0.1, y=0.9, z=0.0, confidence=0.9),
            "left_knee":      Keypoint3D(x=-0.1, y=0.5, z=0.0, confidence=0.9),
            "left_ankle":     Keypoint3D(x=-0.1, y=0.1, z=0.0, confidence=0.9),
            "mid_hip":        Keypoint3D(x=0.0, y=0.9, z=0.0, confidence=0.9),
            "mid_shoulder":   Keypoint3D(x=0.0, y=1.4, z=0.0, confidence=0.9),
            "right_eye":      Keypoint3D(x=0.05, y=1.72, z=0.0, confidence=0.9),
            "left_eye":       Keypoint3D(x=-0.05, y=1.72, z=0.0, confidence=0.9),
        }
    else:
        # Postura inclinada 70°
        kps = {
            "nose":           Keypoint3D(x=0.7, y=1.0, z=0.0, confidence=0.9),
            "neck":           Keypoint3D(x=0.6, y=0.9, z=0.0, confidence=0.9),
            "right_shoulder": Keypoint3D(x=0.7, y=0.85, z=0.0, confidence=0.9),
            "right_elbow":    Keypoint3D(x=0.9, y=0.7, z=0.0, confidence=0.9),
            "right_wrist":    Keypoint3D(x=1.0, y=0.55, z=0.0, confidence=0.9),
            "left_shoulder":  Keypoint3D(x=0.5, y=0.85, z=0.0, confidence=0.9),
            "left_elbow":     Keypoint3D(x=0.3, y=0.7, z=0.0, confidence=0.9),
            "left_wrist":     Keypoint3D(x=0.2, y=0.55, z=0.0, confidence=0.9),
            "right_hip":      Keypoint3D(x=0.15, y=0.9, z=0.0, confidence=0.9),
            "right_knee":     Keypoint3D(x=0.1, y=0.5, z=0.0, confidence=0.9),
            "right_ankle":    Keypoint3D(x=0.1, y=0.1, z=0.0, confidence=0.9),
            "left_hip":       Keypoint3D(x=-0.05, y=0.9, z=0.0, confidence=0.9),
            "left_knee":      Keypoint3D(x=-0.05, y=0.5, z=0.0, confidence=0.9),
            "left_ankle":     Keypoint3D(x=-0.05, y=0.1, z=0.0, confidence=0.9),
            "mid_hip":        Keypoint3D(x=0.05, y=0.9, z=0.0, confidence=0.9),
            "mid_shoulder":   Keypoint3D(x=0.6, y=0.85, z=0.0, confidence=0.9),
            "right_eye":      Keypoint3D(x=0.75, y=1.02, z=0.0, confidence=0.9),
            "left_eye":       Keypoint3D(x=0.65, y=1.02, z=0.0, confidence=0.9),
        }
    return Skeleton3D(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx) / 25.0,
        keypoints=kps,
        scale_px_to_m=0.01,
        person_height_cm=170.0,
    )


@pytest.fixture
def upright_angles():
    return make_upright_angles()


@pytest.fixture
def high_risk_angles():
    return make_high_risk_angles()


@pytest.fixture
def upright_skeleton():
    return make_skeleton(upright=True)


@pytest.fixture
def skeleton_sequence_upright():
    frames = [make_skeleton(i, upright=True) for i in range(5)]
    return SkeletonSequence(
        video_path="/test/video.mp4",
        fps=25.0,
        total_frames=125,
        frames=frames,
    )
