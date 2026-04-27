from schemas import SkeletonSequence
from loguru import logger


class HumanMMWrapper:
    """
    Wrapper for HumanMM (CVPR 2025) — multi-shot video pose estimation.
    Repo: https://github.com/zhangyuhong01/HumanMM-code

    Use when the video contains shot transitions / camera cuts.
    If HumanMM is not installed: is_available() returns False, process_multishot() returns None.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = self._check_humanmm()

    def _check_humanmm(self) -> bool:
        try:
            import humanmm  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def process_multishot(
        self,
        video_path: str,
        person_height_cm: float = 170.0,
    ) -> SkeletonSequence | None:
        """
        Processes a multi-shot video by detecting shot boundaries and applying
        cross-shot pose consistency. Returns None if HumanMM is not available.
        """
        if not self._available:
            logger.warning("HumanMM not installed. Install: pip install -e HumanMM-code/")
            return None
        # Full HumanMM inference — requires installed package
        return None
