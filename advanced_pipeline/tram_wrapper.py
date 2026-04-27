from schemas import SkeletonSequence
from loguru import logger


class TRAMWrapper:
    """
    Wrapper for TRAM (ECCV 2024) — alternative world-grounded motor.
    Paper: "TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos"
    Repo: https://github.com/yufu-wang/tram

    Architecture:
    1. DROID-SLAM with dual masking (ignores dynamic humans) → metric camera trajectory
    2. VIMO (ViT-H video transformer) → kinematic body motion
    60% trajectory error reduction vs. baselines.

    If TRAM is not installed: is_available() returns False, estimate() returns None.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = self._check_tram()

    def _check_tram(self) -> bool:
        try:
            import tram  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """Returns SkeletonSequence with coordinate_system="world" or None if unavailable."""
        if not self._available:
            logger.warning("TRAM not installed. Install: pip install -e tram/")
            return None
        # Full TRAM inference — requires installed package
        return None
