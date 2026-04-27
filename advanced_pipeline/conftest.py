import pytest
import numpy as np
import cv2
from schemas import SkeletonSequence, Skeleton3D, Keypoint3D, KEYPOINT_NAMES, VideoInput
from advanced_pipeline.pipeline_router import VideoProfile, HardwareProfile


def _make_skeleton(frame_idx: int, confidence: float = 0.9) -> Skeleton3D:
    kps = {
        name: Keypoint3D(x=0.0, y=float(frame_idx) * 0.001, z=0.0, confidence=confidence)
        for name in KEYPOINT_NAMES
    }
    return Skeleton3D(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 25.0,
        keypoints=kps,
        scale_px_to_m=0.01,
        person_height_cm=170.0,
        coordinate_system="world",
    )


@pytest.fixture
def simple_skeleton_sequence() -> SkeletonSequence:
    frames = [_make_skeleton(i) for i in range(10)]
    return SkeletonSequence(
        video_path="/fake/video.mp4",
        fps=25.0,
        total_frames=10,
        frames=frames,
    )


@pytest.fixture
def skeleton_sequence_with_occlusion() -> SkeletonSequence:
    """10 frames where frame 5 has very low confidence (occluded)."""
    frames = []
    for i in range(10):
        conf = 0.1 if i == 5 else 0.9
        frames.append(_make_skeleton(i, confidence=conf))
    return SkeletonSequence(
        video_path="/fake/video.mp4",
        fps=25.0,
        total_frames=10,
        frames=frames,
    )


@pytest.fixture
def skeleton_sequence_static_feet() -> SkeletonSequence:
    """10 frames where ankle keypoints don't move (foot in contact)."""
    frames = []
    for i in range(10):
        kps = {name: Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.9) for name in KEYPOINT_NAMES}
        # Override ankles with static position
        kps["left_ankle"] = Keypoint3D(x=0.1, y=0.0, z=0.0, confidence=0.9)
        kps["right_ankle"] = Keypoint3D(x=-0.1, y=0.0, z=0.0, confidence=0.9)
        frames.append(Skeleton3D(
            frame_idx=i,
            timestamp_s=i / 25.0,
            keypoints=kps,
            scale_px_to_m=0.01,
            person_height_cm=170.0,
        ))
    return SkeletonSequence(
        video_path="/fake/video.mp4",
        fps=25.0,
        total_frames=10,
        frames=frames,
    )


@pytest.fixture
def skeleton_sequence_moving_feet() -> SkeletonSequence:
    """10 frames where ankles move significantly each frame."""
    frames = []
    for i in range(10):
        kps = {name: Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.9) for name in KEYPOINT_NAMES}
        kps["left_ankle"] = Keypoint3D(x=float(i) * 0.5, y=0.0, z=0.0, confidence=0.9)
        kps["right_ankle"] = Keypoint3D(x=float(i) * 0.5, y=0.0, z=0.0, confidence=0.9)
        frames.append(Skeleton3D(
            frame_idx=i,
            timestamp_s=i / 25.0,
            keypoints=kps,
            scale_px_to_m=0.01,
            person_height_cm=170.0,
        ))
    return SkeletonSequence(
        video_path="/fake/video.mp4",
        fps=25.0,
        total_frames=10,
        frames=frames,
    )


@pytest.fixture
def cpu_hardware() -> HardwareProfile:
    return HardwareProfile(vram_gb=0.0, cuda_available=False)


@pytest.fixture
def gpu_hardware_4gb() -> HardwareProfile:
    return HardwareProfile(vram_gb=4.0, cuda_available=True)


@pytest.fixture
def synthetic_video_path(tmp_path) -> str:
    path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (640, 480))
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (300, 50), (340, 150), (200, 200, 200), -1)
        cv2.rectangle(frame, (290, 150), (350, 320), (200, 200, 200), -1)
        cv2.rectangle(frame, (290, 320), (315, 430), (200, 200, 200), -1)
        cv2.rectangle(frame, (315, 320), (350, 430), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return str(path)
