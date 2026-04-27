import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pose_pipeline.pose_3d import PoseLifter3D, WINDOW_SIZE, STRIDE
from pose_pipeline.pose_2d import Keypoint2D
from schemas import KEYPOINT_NAMES


def make_kps_seq(n_frames: int) -> list[dict[str, Keypoint2D]]:
    return [
        {name: Keypoint2D(x=320.0, y=240.0, confidence=0.9) for name in KEYPOINT_NAMES}
        for _ in range(n_frames)
    ]


def make_mock_session():
    mock_session = MagicMock()
    output = np.zeros((1, WINDOW_SIZE, 17, 3), dtype=np.float32)
    mock_session.run.return_value = [output]
    mock_session.get_inputs.return_value = [MagicMock(name="input_2d")]
    return mock_session


def test_window_size_constant():
    assert WINDOW_SIZE == 243


def test_stride_constant():
    assert STRIDE == 121


def test_lifter_init_no_model(tmp_path):
    fake_path = str(tmp_path / "mb.onnx")
    lifter = PoseLifter3D(fake_path, device="cpu")
    assert lifter is not None


def test_lift_short_sequence(tmp_path):
    """lift() con secuencia corta retorna lista del mismo tamaño."""
    fake_path = str(tmp_path / "mb.onnx")
    open(fake_path, "w").close()
    n_frames = 10
    seq = make_kps_seq(n_frames)

    with patch("onnxruntime.InferenceSession", return_value=make_mock_session()):
        lifter = PoseLifter3D(fake_path, device="cpu")
        result = lifter.lift(seq)

    assert len(result) == n_frames
    for frame_kps in result:
        assert isinstance(frame_kps, dict)
        for name in KEYPOINT_NAMES:
            assert name in frame_kps


def test_lift_full_window(tmp_path):
    """lift() con exactamente WINDOW_SIZE frames."""
    fake_path = str(tmp_path / "mb.onnx")
    open(fake_path, "w").close()
    seq = make_kps_seq(WINDOW_SIZE)

    with patch("onnxruntime.InferenceSession", return_value=make_mock_session()):
        lifter = PoseLifter3D(fake_path, device="cpu")
        result = lifter.lift(seq)

    assert len(result) == WINDOW_SIZE


def test_lift_empty_sequence(tmp_path):
    fake_path = str(tmp_path / "mb.onnx")
    open(fake_path, "w").close()

    with patch("onnxruntime.InferenceSession", return_value=make_mock_session()):
        lifter = PoseLifter3D(fake_path, device="cpu")
        result = lifter.lift([])

    assert result == []


def test_lift_no_model_raises(tmp_path):
    """Sin modelo, lift() lanza RuntimeError."""
    fake_path = str(tmp_path / "nonexistent.onnx")
    lifter = PoseLifter3D(fake_path, device="cpu")
    seq = make_kps_seq(5)
    with pytest.raises(RuntimeError):
        lifter.lift(seq)
