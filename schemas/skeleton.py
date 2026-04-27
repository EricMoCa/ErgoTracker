from pydantic import BaseModel, Field


class Keypoint3D(BaseModel):
    x: float
    y: float
    z: float
    confidence: float = Field(ge=0.0, le=1.0)
    occluded: bool = False


KEYPOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "mid_hip", "mid_shoulder",
    "right_eye", "left_eye",
]


class Skeleton3D(BaseModel):
    frame_idx: int
    timestamp_s: float
    keypoints: dict[str, Keypoint3D]
    scale_px_to_m: float
    person_height_cm: float
    coordinate_system: str = "world"


class SkeletonSequence(BaseModel):
    video_path: str
    fps: float
    total_frames: int
    frames: list[Skeleton3D]
