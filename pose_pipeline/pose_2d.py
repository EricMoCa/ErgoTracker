"""
pose_pipeline/pose_2d.py
Estimación de keypoints 2D usando RTMPose-m ONNX.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from schemas import KEYPOINT_NAMES
from schemas.skeleton import Keypoint3D  # Keypoint2D se define aquí localmente

from .detector import BBox

# ---------------------------------------------------------------------------
# Definición local de Keypoint2D (no está en schemas — es interno)
# ---------------------------------------------------------------------------
class Keypoint2D:
    """Keypoint 2D con coordenadas en píxeles y confianza."""

    __slots__ = ("x", "y", "confidence")

    def __init__(self, x: float, y: float, confidence: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.confidence = float(confidence)

    def __repr__(self) -> str:
        return f"Keypoint2D(x={self.x:.1f}, y={self.y:.1f}, conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# Mapeo RTMPose (17 puntos COCO) → KEYPOINT_NAMES de schemas
# ---------------------------------------------------------------------------
# RTMPose-m (body7) produce 17 keypoints en orden COCO:
#  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

_COCO_IDX_TO_NAME: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Tamaño de crop para RTMPose
_INPUT_SIZE = (192, 256)  # (W, H)


def _make_empty_keypoints() -> dict[str, Keypoint2D]:
    """Retorna todos los keypoints con confianza 0 en el origen."""
    return {name: Keypoint2D(0.0, 0.0, 0.0) for name in KEYPOINT_NAMES}


def _crop_and_resize(
    frame: np.ndarray, bbox: BBox, target_size: tuple[int, int] = _INPUT_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta y redimensiona el frame al bbox.

    Returns:
        (img_crop, affine_params) — affine_params = [scale_x, scale_y, x1, y1]
    """
    import cv2

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return crop, np.array([1.0, 1.0, 0.0, 0.0])

    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    scale_x = crop_w / target_size[0]
    scale_y = crop_h / target_size[1]
    affine = np.array([scale_x, scale_y, float(x1), float(y1)])
    return resized, affine


def _preprocess_rtmpose(img: np.ndarray) -> np.ndarray:
    """Preprocesa la imagen para RTMPose: normalizar y transponer."""
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_f = img.astype(np.float32)
    img_f = (img_f - mean) / std
    img_f = img_f.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img_f, 0)  # (1, 3, H, W)


def _postprocess_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    affine: np.ndarray,
    img_size: tuple[int, int] = _INPUT_SIZE,
) -> dict[str, Keypoint2D]:
    """
    Postprocesa salida SimCC de RTMPose.
    simcc_x: (1, num_kpts, W*simcc_split)
    simcc_y: (1, num_kpts, H*simcc_split)
    """
    simcc_split = 2  # RTMPose usa split_ratio=2
    # (num_kpts, W*2) y (num_kpts, H*2)
    sx = simcc_x[0]  # (num_kpts, W*2)
    sy = simcc_y[0]  # (num_kpts, H*2)

    # Obtener máximos
    x_indices = np.argmax(sx, axis=1).astype(np.float32) / simcc_split
    y_indices = np.argmax(sy, axis=1).astype(np.float32) / simcc_split
    x_scores = np.max(sx, axis=1)
    y_scores = np.max(sy, axis=1)
    scores = (x_scores + y_scores) / 2.0

    # Normalizar scores a [0,1] con softmax-like
    scores = 1.0 / (1.0 + np.exp(-scores))

    # Mapear a coordenadas originales usando affine
    scale_x, scale_y, off_x, off_y = affine
    x_coords = x_indices * scale_x + off_x
    y_coords = y_indices * scale_y + off_y

    result = _make_empty_keypoints()
    for coco_idx, name in _COCO_IDX_TO_NAME.items():
        if coco_idx < len(x_coords):
            result[name] = Keypoint2D(
                x=float(x_coords[coco_idx]),
                y=float(y_coords[coco_idx]),
                confidence=float(np.clip(scores[coco_idx], 0.0, 1.0)),
            )

    # Derivar keypoints que no están en COCO directamente
    _derive_extra_keypoints(result)
    return result


def _derive_extra_keypoints(kpts: dict[str, Keypoint2D]) -> None:
    """Deriva keypoints adicionales (neck, mid_hip, mid_shoulder) por promedio."""
    # neck = promedio de hombros
    if "left_shoulder" in kpts and "right_shoulder" in kpts:
        ls, rs = kpts["left_shoulder"], kpts["right_shoulder"]
        conf = (ls.confidence + rs.confidence) / 2.0
        kpts["neck"] = Keypoint2D(
            (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0, conf
        )

    # mid_hip = promedio de caderas
    if "left_hip" in kpts and "right_hip" in kpts:
        lh, rh = kpts["left_hip"], kpts["right_hip"]
        conf = (lh.confidence + rh.confidence) / 2.0
        kpts["mid_hip"] = Keypoint2D(
            (lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0, conf
        )

    # mid_shoulder = mismo que neck
    if "neck" in kpts:
        n = kpts["neck"]
        kpts["mid_shoulder"] = Keypoint2D(n.x, n.y, n.confidence)


class PoseEstimator2D:
    """Estimación de keypoints 2D usando RTMPose-m ONNX."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """
        Args:
            model_path: ruta al archivo ONNX de RTMPose-m.
            device: 'cpu' o 'cuda'.
        """
        self.model_path = str(model_path)
        self.device = device
        self._session = None
        logger.debug(f"PoseEstimator2D inicializado con modelo: {self.model_path}")

    def _get_session(self):
        """Lazy-load de la sesión ONNX."""
        if self._session is None:
            try:
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self._session = ort.InferenceSession(
                    self.model_path, providers=providers
                )
                logger.debug("ONNX session de pose 2D cargada")
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo cargar el modelo ONNX '{self.model_path}': {e}"
                ) from e
        return self._session

    def estimate(
        self, frame: np.ndarray, bbox: BBox
    ) -> dict[str, Keypoint2D]:
        """
        Estima keypoints 2D en un frame dado un bounding box.

        Args:
            frame: imagen BGR completa (H, W, 3).
            bbox: [x1, y1, x2, y2, confidence].

        Returns:
            dict nombre_keypoint -> Keypoint2D con todos los KEYPOINT_NAMES.
        """
        session = self._get_session()
        crop, affine = _crop_and_resize(frame, bbox)
        inp = _preprocess_rtmpose(crop)

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: inp})

        # RTMPose SimCC produce dos outputs: simcc_x, simcc_y
        if len(outputs) >= 2:
            simcc_x, simcc_y = outputs[0], outputs[1]
            return _postprocess_simcc(simcc_x, simcc_y, affine)
        else:
            # Fallback: output único tipo heatmap/coordenadas directas
            logger.warning("Output inesperado de RTMPose, usando fallback")
            return _make_empty_keypoints()

    def estimate_batch(
        self,
        frames: list[np.ndarray],
        bboxes: list[BBox],
    ) -> list[dict[str, Keypoint2D]]:
        """
        Procesa múltiples frames en batch.

        Args:
            frames: lista de imágenes BGR.
            bboxes: lista de bounding boxes correspondientes.

        Returns:
            Lista de dicts de keypoints por frame.
        """
        if len(frames) != len(bboxes):
            raise ValueError(
                f"Número de frames ({len(frames)}) != número de bboxes ({len(bboxes)})"
            )
        results = []
        for frame, bbox in zip(frames, bboxes):
            results.append(self.estimate(frame, bbox))
        return results
