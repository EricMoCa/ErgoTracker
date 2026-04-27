"""
pose_pipeline/detector.py
Detección de personas usando YOLOv8n ONNX.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# BBox: [x1, y1, x2, y2, confidence]
BBox = list[float]

# Índices de clase COCO para "person"
_PERSON_CLASS_ID = 0

# Umbral de confianza por defecto
_CONF_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.45

# Tamaño de entrada YOLOv8n
_INPUT_SIZE = (640, 640)


def _letterbox(
    img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona manteniendo aspecto con padding."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_w) / 2
    dh = (new_shape[0] - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    import cv2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_padded, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression simple."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class PersonDetector:
    """Detecta bounding boxes de personas usando YOLOv8n ONNX."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """
        Args:
            model_path: ruta al archivo ONNX de YOLOv8n.
            device: 'cpu' o 'cuda'.
        """
        self.model_path = str(model_path)
        self.device = device
        self._session = None
        logger.debug(f"PersonDetector inicializado con modelo: {self.model_path}")

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
                logger.debug("ONNX session de detección cargada")
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo cargar el modelo ONNX '{self.model_path}': {e}"
                ) from e
        return self._session

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Preprocesa el frame para YOLOv8n."""
        img, ratio, pad = _letterbox(frame, _INPUT_SIZE)
        # BGR -> RGB, HWC -> CHW, normalizar [0,1]
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # (1, 3, H, W)
        return img, ratio, pad

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[int, int],
        orig_h: int,
        orig_w: int,
    ) -> list[BBox]:
        """
        Postprocesa la salida de YOLOv8n.
        YOLOv8 ONNX output shape: (1, 84, 8400) — [x,y,w,h, 80 clases]
        """
        # output shape: (1, num_classes+4, num_anchors)
        pred = output[0]  # (84, 8400)
        pred = pred.T  # (8400, 84)

        # Extraer boxes y scores
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]  # (8400, 80)

        # Solo clase persona (índice 0)
        person_scores = scores_all[:, _PERSON_CLASS_ID]
        mask = person_scores > _CONF_THRESHOLD

        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]

        # xywh -> xyxy
        x_c, y_c, w, h = (
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        )
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2

        # Deshacer letterbox
        pad_x, pad_y = pad
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        # Clip a dimensiones originales
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, person_scores, _IOU_THRESHOLD)

        result: list[BBox] = []
        for idx in keep:
            result.append(
                [
                    float(boxes[idx, 0]),
                    float(boxes[idx, 1]),
                    float(boxes[idx, 2]),
                    float(boxes[idx, 3]),
                    float(person_scores[idx]),
                ]
            )
        return result

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """
        Detecta personas en un frame.

        Args:
            frame: imagen BGR (H, W, 3).

        Returns:
            Lista de BBox [x1, y1, x2, y2, confidence].
        """
        orig_h, orig_w = frame.shape[:2]
        session = self._get_session()
        inp, ratio, pad = self._preprocess(frame)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})
        return self._postprocess(outputs[0], ratio, pad, orig_h, orig_w)

    def detect_primary(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Retorna el BBox de la persona con mayor área (operador principal).

        Args:
            frame: imagen BGR (H, W, 3).

        Returns:
            BBox [x1, y1, x2, y2, confidence] o None si no se detecta nadie.
        """
        detections = self.detect(frame)
        if not detections:
            return None

        def area(bbox: BBox) -> float:
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return max(detections, key=area)
