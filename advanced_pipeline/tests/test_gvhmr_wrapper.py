import pytest
from advanced_pipeline.gvhmr_wrapper import GVHMRWrapper


def test_gvhmr_is_available_returns_bool():
    wrapper = GVHMRWrapper(device="cpu")
    assert isinstance(wrapper.is_available(), bool)


def test_gvhmr_not_available_without_package():
    wrapper = GVHMRWrapper(device="cpu")
    # GVHMR package is not installed in CI — expect False
    assert wrapper.is_available() is False


def test_gvhmr_estimate_returns_none_when_not_installed(tmp_path):
    fake_video = str(tmp_path / "video.mp4")
    open(fake_video, "w").close()
    wrapper = GVHMRWrapper(device="cpu")
    result = wrapper.estimate(fake_video, person_height_cm=170.0)
    assert result is None


def test_gvhmr_estimate_signature_accepts_height(tmp_path):
    fake_video = str(tmp_path / "video.mp4")
    open(fake_video, "w").close()
    wrapper = GVHMRWrapper(device="cpu")
    # Should not raise even if video is empty
    result = wrapper.estimate(fake_video, person_height_cm=165.0)
    assert result is None


def test_gvhmr_has_visual_odometry():
    wrapper = GVHMRWrapper(device="cpu")
    from advanced_pipeline.visual_odometry import VisualOdometry
    assert isinstance(wrapper.vo, VisualOdometry)
