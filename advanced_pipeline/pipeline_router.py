from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class VideoProfile:
    duration_s: float
    has_multiple_shots: bool = False
    camera_motion_score: float = 0.0
    occlusion_score: float = 0.0
    requires_gait_analysis: bool = False


@dataclass
class HardwareProfile:
    vram_gb: float = 0.0
    cuda_available: bool = False

    @classmethod
    def detect(cls) -> "HardwareProfile":
        try:
            import torch
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                return cls(vram_gb=free_vram, cuda_available=True)
        except Exception:
            pass
        return cls(vram_gb=0.0, cuda_available=False)


class PipelineRouter:
    """
    Selects the best available world-grounded motor for the given video and hardware.

    Priority:
    1. HumanMM  → multi-shot videos
    2. GVHMR    → long videos (>5 min), moving camera, default GPU
    3. GVHMR + contact_refinement → gait analysis
    4. WHAM     → gait analysis on short video (<2 min), no GVHMR
    5. TRAM     → GVHMR unavailable, GPU >= 4 GB
    6. MotionBERT Lite → CPU fallback (always available)
    """

    def select(
        self,
        video_profile: VideoProfile,
        hardware: Optional[HardwareProfile] = None,
        gvhmr_available: bool = False,
        wham_available: bool = False,
        tram_available: bool = False,
        humanmm_available: bool = False,
    ) -> str:
        hw = hardware or HardwareProfile.detect()

        if video_profile.has_multiple_shots and humanmm_available:
            logger.info("Router → HumanMM (multi-shot video)")
            return "humanmm"

        if hw.cuda_available and hw.vram_gb >= 3.0:
            if gvhmr_available:
                if video_profile.requires_gait_analysis:
                    logger.info("Router → GVHMR + contact_refinement (gait analysis)")
                    return "gvhmr_with_contact"
                if video_profile.duration_s > 300 or video_profile.camera_motion_score > 0.2:
                    logger.info("Router → GVHMR (long video / moving camera)")
                    return "gvhmr"
                logger.info("Router → GVHMR (default GPU)")
                return "gvhmr"

            if wham_available and video_profile.requires_gait_analysis and video_profile.duration_s < 120:
                logger.info("Router → WHAM (gait analysis, short video)")
                return "wham"

            if tram_available and hw.vram_gb >= 4.0:
                logger.info("Router → TRAM (GVHMR unavailable)")
                return "tram"

        logger.warning("Router → MotionBERT Lite (CPU fallback)")
        return "motionbert_lite"
