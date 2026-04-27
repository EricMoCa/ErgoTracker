import pytest
from unittest.mock import patch, MagicMock
from schemas import SkeletonSequence, VideoInput
from advanced_pipeline.pipeline import AdvancedPosePipeline
from advanced_pipeline.pipeline_router import HardwareProfile


def test_output_always_skeleton_sequence(synthetic_video_path):
    """Pipeline always returns SkeletonSequence even without GPU models."""
    pipeline = AdvancedPosePipeline(device="cpu")
    video_input = VideoInput(path=synthetic_video_path, person_height_cm=170.0)
    result = pipeline.process(video_input)
    assert isinstance(result, SkeletonSequence)


def test_fallback_without_gpu(synthetic_video_path):
    """Without CUDA, falls back to base PosePipeline."""
    with patch(
        "advanced_pipeline.pipeline.HardwareProfile.detect",
        return_value=HardwareProfile(vram_gb=0.0, cuda_available=False),
    ):
        pipeline = AdvancedPosePipeline(device="cpu")
        video_input = VideoInput(path=synthetic_video_path)
        result = pipeline.process(video_input)
    assert isinstance(result, SkeletonSequence)


def test_fallback_returns_correct_video_path(synthetic_video_path):
    pipeline = AdvancedPosePipeline(device="cpu")
    video_input = VideoInput(path=synthetic_video_path)
    result = pipeline.process(video_input)
    assert result.video_path == synthetic_video_path


def test_process_nonexistent_video_raises():
    pipeline = AdvancedPosePipeline(device="cpu")
    with pytest.raises(FileNotFoundError):
        pipeline.process(VideoInput(path="/nonexistent/video.mp4"))


def test_process_video_path_shortcut(synthetic_video_path):
    pipeline = AdvancedPosePipeline(device="cpu")
    result = pipeline.process_video_path(synthetic_video_path, person_height_cm=170.0)
    assert isinstance(result, SkeletonSequence)


def test_skeleton_has_frames(synthetic_video_path):
    pipeline = AdvancedPosePipeline(device="cpu")
    result = pipeline.process(VideoInput(path=synthetic_video_path))
    assert len(result.frames) > 0


def test_gvhmr_result_used_when_available(synthetic_video_path):
    """When GVHMR returns a result, it should be returned directly."""
    from unittest.mock import MagicMock
    from schemas import Skeleton3D, Keypoint3D, KEYPOINT_NAMES
    fake_seq = SkeletonSequence(
        video_path=synthetic_video_path,
        fps=25.0,
        total_frames=1,
        frames=[Skeleton3D(
            frame_idx=0,
            timestamp_s=0.0,
            keypoints={n: Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.9) for n in KEYPOINT_NAMES},
            scale_px_to_m=0.01,
            person_height_cm=170.0,
            coordinate_system="world",
        )],
    )

    with patch("advanced_pipeline.pipeline.HardwareProfile.detect",
               return_value=HardwareProfile(vram_gb=4.0, cuda_available=True)):
        pipeline = AdvancedPosePipeline(device="cpu")

    mock_gvhmr = MagicMock()
    mock_gvhmr.is_available.return_value = True
    mock_gvhmr.estimate.return_value = fake_seq
    pipeline._gvhmr = mock_gvhmr

    result = pipeline.process(VideoInput(path=synthetic_video_path))
    assert isinstance(result, SkeletonSequence)


def test_build_video_profile_returns_defaults(synthetic_video_path):
    pipeline = AdvancedPosePipeline(device="cpu")
    video_input = VideoInput(path=synthetic_video_path)
    profile = pipeline._build_video_profile(video_input)
    assert profile.duration_s == 0.0
    assert profile.has_multiple_shots is False
    assert profile.occlusion_score == 0.0
