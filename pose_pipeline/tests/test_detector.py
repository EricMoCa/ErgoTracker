import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pose_pipeline.detector import PersonDetector, BBox, _nms, _letterbox


def test_letterbox_output_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result, ratio, padding = _letterbox(img, (640, 640))
    assert result.shape == (640, 640, 3)


def test_nms_empty():
    boxes = np.array([]).reshape(0, 4)
    scores = np.array([])
    result = _nms(boxes, scores, 0.45)
    assert result == []


def test_nms_single_box():
    boxes = np.array([[0, 0, 10, 10]])
    scores = np.array([0.9])
    result = _nms(boxes, scores, 0.45)
    assert result == [0]


def test_nms_removes_overlapping():
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [100, 100, 110, 110],
    ])
    scores = np.array([0.9, 0.8, 0.7])
    result = _nms(boxes, scores, 0.45)
    assert 0 in result
    assert 2 in result


def test_detector_init_no_model(tmp_path):
    """PersonDetector se instancia sin error aunque el modelo no exista."""
    fake_path = str(tmp_path / "nonexistent.onnx")
    detector = PersonDetector(fake_path, device="cpu")
    assert detector is not None


def test_detect_primary_mock(tmp_path):
    """detect_primary retorna None o BBox válido con sesión mockeada."""
    fake_path = str(tmp_path / "model.onnx")
    open(fake_path, "w").close()

    mock_session = MagicMock()
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    mock_session.run.return_value = [output]
    mock_session.get_inputs.return_value = [MagicMock(name="images")]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        detector = PersonDetector(fake_path, device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect_primary(frame)
        assert bbox is None or (isinstance(bbox, list) and len(bbox) == 5)


def test_detect_primary_no_model_raises(tmp_path):
    """Sin modelo, detect_primary lanza RuntimeError."""
    fake_path = str(tmp_path / "nonexistent.onnx")
    detector = PersonDetector(fake_path, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        detector.detect_primary(frame)
