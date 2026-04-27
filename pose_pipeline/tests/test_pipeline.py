import pytest
from unittest.mock import patch, MagicMock
from schemas import VideoInput, SkeletonSequence, KEYPOINT_NAMES
from pose_pipeline import PosePipeline


def test_process_returns_skeleton_sequence(synthetic_video_path):
    """Con video sintético y sin modelos, retorna SkeletonSequence."""
    pipeline = PosePipeline(device="cpu")
    video_input = VideoInput(path=synthetic_video_path, person_height_cm=170.0)
    result = pipeline.process(video_input)
    assert isinstance(result, SkeletonSequence)


def test_skeleton_has_correct_keypoints(synthetic_video_path):
    """Todos los KEYPOINT_NAMES presentes en cada frame."""
    pipeline = PosePipeline(device="cpu")
    video_input = VideoInput(path=synthetic_video_path, person_height_cm=170.0)
    result = pipeline.process(video_input)
    assert len(result.frames) > 0
    for frame in result.frames:
        for name in KEYPOINT_NAMES:
            assert name in frame.keypoints


def test_sample_rate_reduces_frames(synthetic_video_path):
    """fps_sample_rate=2 analiza la mitad de frames que sample_rate=1."""
    pipeline = PosePipeline(device="cpu")
    result_full = pipeline.process(
        VideoInput(path=synthetic_video_path, fps_sample_rate=1)
    )
    result_half = pipeline.process(
        VideoInput(path=synthetic_video_path, fps_sample_rate=2)
    )
    assert len(result_full.frames) > len(result_half.frames)


def test_process_video_path_shortcut(synthetic_video_path):
    """process_video_path() retorna SkeletonSequence."""
    pipeline = PosePipeline(device="cpu")
    result = pipeline.process_video_path(synthetic_video_path, person_height_cm=170.0)
    assert isinstance(result, SkeletonSequence)


def test_skeleton_sequence_video_path(synthetic_video_path):
    pipeline = PosePipeline(device="cpu")
    result = pipeline.process(VideoInput(path=synthetic_video_path))
    assert result.video_path == synthetic_video_path


def test_height_in_skeleton(synthetic_video_path):
    """Con altura 170cm, el rango Y del skeleton debe ser ~[0, 1.7]."""
    pipeline = PosePipeline(device="cpu")
    result = pipeline.process(VideoInput(path=synthetic_video_path, person_height_cm=170.0))
    if result.frames:
        frame = result.frames[0]
        nose_y = frame.keypoints["nose"].y
        ankle_y = frame.keypoints["left_ankle"].y
        height = abs(nose_y - ankle_y)
        # Con modo sintético la altura debe estar cerca de 1.7m
        assert 1.0 < height < 2.5


def test_process_nonexistent_video():
    pipeline = PosePipeline(device="cpu")
    with pytest.raises(FileNotFoundError):
        pipeline.process(VideoInput(path="/nonexistent/video.mp4"))
