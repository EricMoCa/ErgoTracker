import pytest
import cv2
import numpy as np


@pytest.fixture
def synthetic_video_path(tmp_path):
    """Video sintético de 2 segundos a 25fps con un rectángulo (figura humana simple)."""
    path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (640, 480))
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dibujar figura humana simple
        cv2.rectangle(frame, (300, 50), (340, 150), (200, 200, 200), -1)  # cabeza
        cv2.rectangle(frame, (290, 150), (350, 320), (200, 200, 200), -1)  # torso
        cv2.rectangle(frame, (290, 320), (315, 430), (200, 200, 200), -1)  # pierna izq
        cv2.rectangle(frame, (315, 320), (350, 430), (200, 200, 200), -1)  # pierna der
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def mock_keypoints_2d():
    """243 frames de keypoints 2D sintéticos para tests de PoseLifter3D."""
    from pose_pipeline.pose_2d import Keypoint2D
    from schemas import KEYPOINT_NAMES
    frames = []
    for i in range(243):
        kps = {name: Keypoint2D(x=320.0, y=240.0, confidence=0.9) for name in KEYPOINT_NAMES}
        frames.append(kps)
    return frames
