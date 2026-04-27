import pytest
from advanced_pipeline.pipeline_router import PipelineRouter, VideoProfile, HardwareProfile


@pytest.fixture
def router():
    return PipelineRouter()


@pytest.fixture
def cpu_hw():
    return HardwareProfile(vram_gb=0.0, cuda_available=False)


@pytest.fixture
def gpu_hw_4gb():
    return HardwareProfile(vram_gb=4.0, cuda_available=True)


@pytest.fixture
def gpu_hw_3gb():
    return HardwareProfile(vram_gb=3.0, cuda_available=True)


def test_router_fallback_no_gpu(router, cpu_hw):
    profile = VideoProfile(duration_s=60.0)
    result = router.select(profile, hardware=cpu_hw)
    assert result == "motionbert_lite"


def test_router_fallback_no_models_with_gpu(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=False, wham_available=False,
        tram_available=False, humanmm_available=False,
    )
    assert result == "motionbert_lite"


def test_router_selects_gvhmr_default_gpu(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0)
    result = router.select(profile, hardware=gpu_hw_4gb, gvhmr_available=True)
    assert result == "gvhmr"


def test_router_selects_gvhmr_for_long_video(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=400.0)
    result = router.select(profile, hardware=gpu_hw_4gb, gvhmr_available=True)
    assert result == "gvhmr"


def test_router_selects_gvhmr_for_moving_camera(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=30.0, camera_motion_score=0.5)
    result = router.select(profile, hardware=gpu_hw_4gb, gvhmr_available=True)
    assert result == "gvhmr"


def test_router_selects_gvhmr_with_contact_for_gait(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0, requires_gait_analysis=True)
    result = router.select(profile, hardware=gpu_hw_4gb, gvhmr_available=True)
    assert result == "gvhmr_with_contact"


def test_router_selects_humanmm_for_multishot(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0, has_multiple_shots=True)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=True, humanmm_available=True,
    )
    assert result == "humanmm"


def test_router_humanmm_requires_humanmm_available(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0, has_multiple_shots=True)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=True, humanmm_available=False,
    )
    # No HumanMM → falls back to GVHMR
    assert result == "gvhmr"


def test_router_selects_tram_when_gvhmr_unavailable(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=60.0)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=False, tram_available=True,
    )
    assert result == "tram"


def test_router_tram_requires_4gb(router):
    hw_small = HardwareProfile(vram_gb=3.5, cuda_available=True)
    profile = VideoProfile(duration_s=60.0)
    result = router.select(
        profile, hardware=hw_small,
        gvhmr_available=False, tram_available=True,
    )
    assert result == "motionbert_lite"


def test_router_selects_wham_for_gait_short_video(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=90.0, requires_gait_analysis=True)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=False, wham_available=True,
    )
    assert result == "wham"


def test_router_wham_not_for_long_video(router, gpu_hw_4gb):
    profile = VideoProfile(duration_s=200.0, requires_gait_analysis=True)
    result = router.select(
        profile, hardware=gpu_hw_4gb,
        gvhmr_available=False, wham_available=True,
    )
    # duration >= 120s → WHAM not selected
    assert result != "wham"


def test_hardware_profile_detect_returns_instance():
    hw = HardwareProfile.detect()
    assert isinstance(hw, HardwareProfile)
    assert hw.vram_gb >= 0.0


def test_video_profile_defaults():
    vp = VideoProfile(duration_s=10.0)
    assert vp.has_multiple_shots is False
    assert vp.camera_motion_score == 0.0
    assert vp.occlusion_score == 0.0
    assert vp.requires_gait_analysis is False
