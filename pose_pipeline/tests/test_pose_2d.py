import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pose_pipeline.pose_2d import PoseEstimator2D, Keypoint2D
from schemas import KEYPOINT_NAMES


def test_keypoint2d_creation():
    kp = Keypoint2D(x=100.0, y=200.0, confidence=0.85)
    assert kp.x == 100.0
    assert kp.y == 200.0
    assert kp.confidence == 0.85


def test_pose_estimator_init_no_model(tmp_path):
    fake_path = str(tmp_path / "rtmpose.onnx")
    estimator = PoseEstimator2D(fake_path, device="cpu")
    assert estimator is not None


def _make_mock_session():
    mock_session = MagicMock()
    simcd_x = np.zeros((1, 17, 192), dtype=np.float32)
    simcd_y = np.zeros((1, 17, 256), dtype=np.float32)
    simcd_x[0, :, 96] = 1.0
    simcd_y[0, :, 128] = 1.0
    mock_session.run.return_value = [simcd_x, simcd_y]
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    return mock_session


def test_estimate_returns_all_keypoints(tmp_path):
    """estimate() retorna keypoints para todos los KEYPOINT_NAMES."""
    fake_path = str(tmp_path / "rtmpose.onnx")
    open(fake_path, "w").close()

    with patch("onnxruntime.InferenceSession", return_value=_make_mock_session()):
        estimator = PoseEstimator2D(fake_path, device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [100.0, 50.0, 400.0, 450.0, 0.9]
        result = estimator.estimate(frame, bbox)

    assert isinstance(result, dict)
    for name in KEYPOINT_NAMES:
        assert name in result
        assert isinstance(result[name], Keypoint2D)


def test_estimate_batch_returns_list(tmp_path):
    fake_path = str(tmp_path / "rtmpose.onnx")
    open(fake_path, "w").close()

    with patch("onnxruntime.InferenceSession", return_value=_make_mock_session()):
        estimator = PoseEstimator2D(fake_path, device="cpu")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        bboxes = [[0.0, 0.0, 640.0, 480.0, 1.0]] * 3
        result = estimator.estimate_batch(frames, bboxes)

    assert len(result) == 3


def test_estimate_no_model_raises(tmp_path):
    """Sin modelo, estimate() lanza RuntimeError."""
    fake_path = str(tmp_path / "nonexistent.onnx")
    estimator = PoseEstimator2D(fake_path, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [0.0, 0.0, 640.0, 480.0, 1.0]
    with pytest.raises(RuntimeError):
        estimator.estimate(frame, bbox)
