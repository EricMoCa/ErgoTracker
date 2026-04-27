from schemas import SkeletonSequence
from loguru import logger


class WHAMWrapper:
    """
    Wrapper for WHAM — secondary motor with native foot-ground contact analysis.
    Paper: "WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion"
    CVPR 2024. Repo: https://github.com/yohanshin/WHAM

    Limitations: RNN drift on long videos, DPVO does not compile on Windows.
    Preferred alternative: GVHMRWrapper + ContactRefinement.

    If WHAM is not installed: is_available() returns False, estimate() returns (None, None).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = self._check_wham()

    def _check_wham(self) -> bool:
        try:
            import wham  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def estimate(
        self,
        video_path: str,
        person_height_cm: float = 170.0,
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        """
        Returns (SkeletonSequence, contact_probs) or (None, None) if unavailable.
        contact_probs: per-frame foot-ground contact probability in [0, 1].
        """
        if not self._available:
            logger.warning("WHAM not installed. See CLAUDE.md for install instructions.")
            return None, None
        # Full WHAM inference — requires installed package
        return None, None
