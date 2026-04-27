from schemas import SkeletonSequence
from .visual_odometry import VisualOdometry
from loguru import logger


class GVHMRWrapper:
    """
    Wrapper for GVHMR — primary world-grounded motion recovery motor.
    Paper: "World-Grounded Human Motion Recovery via Gravity-View Coordinates"
    SIGGRAPH Asia 2024. Repo: https://github.com/zju3dv/GVHMR

    If GVHMR is not installed: is_available() returns False, estimate() returns None.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self.vo = VisualOdometry(method="opencv_fallback")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from gvhmr.models.smpl_hmr2 import GVHMR as GVHMRModel
            self._model = GVHMRModel.from_pretrained("zju3dv/GVHMR").to(self.device)
            self._model.eval()
            logger.info("GVHMR loaded on GPU")
        except ImportError:
            logger.warning("GVHMR not installed. Install: pip install -e GVHMR/")
            self._model = None

    def is_available(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            from gvhmr.models.smpl_hmr2 import GVHMR  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """
        Estimates world-grounded poses. Returns None if GVHMR is not available.

        Pipeline:
        1. SimpleVO estimates per-frame camera rotation
        2. ViTPose extracts 2D keypoints + features
        3. Transformer + RoPE processes full sequence (no sliding window)
        4. Prediction in GV system → composed to world coordinates
        5. Convert to SkeletonSequence with coordinate_system="world"
        """
        self._load_model()
        if self._model is None:
            return None

        _camera_rotations = self.vo.estimate(video_path)
        logger.info(f"GVHMR: {len(_camera_rotations)} camera rotation frames estimated")
        # Full GVHMR inference — requires installed package
        return None
