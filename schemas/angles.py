from pydantic import BaseModel
from typing import Optional


class JointAngles(BaseModel):
    """Ángulos articulares en grados, calculados desde Skeleton3D."""
    frame_idx: int
    timestamp_s: float

    trunk_flexion: Optional[float] = None
    trunk_lateral_bending: Optional[float] = None
    trunk_rotation: Optional[float] = None

    neck_flexion: Optional[float] = None
    neck_lateral_bending: Optional[float] = None

    shoulder_elevation_left: Optional[float] = None
    shoulder_elevation_right: Optional[float] = None
    shoulder_abduction_left: Optional[float] = None
    shoulder_abduction_right: Optional[float] = None

    elbow_flexion_left: Optional[float] = None
    elbow_flexion_right: Optional[float] = None

    wrist_flexion_left: Optional[float] = None
    wrist_flexion_right: Optional[float] = None
    wrist_deviation_left: Optional[float] = None
    wrist_deviation_right: Optional[float] = None

    hip_flexion_left: Optional[float] = None
    hip_flexion_right: Optional[float] = None
    knee_flexion_left: Optional[float] = None
    knee_flexion_right: Optional[float] = None
