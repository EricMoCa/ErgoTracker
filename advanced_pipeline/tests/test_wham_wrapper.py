import pytest
from advanced_pipeline.wham_wrapper import WHAMWrapper


def test_wham_is_available_returns_bool():
    wrapper = WHAMWrapper(device="cpu")
    assert isinstance(wrapper.is_available(), bool)


def test_wham_not_available_without_package():
    wrapper = WHAMWrapper(device="cpu")
    assert wrapper.is_available() is False


def test_wham_estimate_returns_none_tuple_when_not_installed():
    wrapper = WHAMWrapper(device="cpu")
    seq, contact_probs = wrapper.estimate("/fake/video.mp4", person_height_cm=170.0)
    assert seq is None
    assert contact_probs is None


def test_wham_estimate_returns_two_element_tuple():
    wrapper = WHAMWrapper(device="cpu")
    result = wrapper.estimate("/fake/video.mp4")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_wham_default_height():
    wrapper = WHAMWrapper(device="cpu")
    seq, _ = wrapper.estimate("/fake/video.mp4")
    assert seq is None
