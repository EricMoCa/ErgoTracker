"""
pose_pipeline/pose_3d.py
Lifting 2D->3D usando MotionBERT Lite ONNX.
Procesa ventanas de 243 frames con stride de 121.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from schemas import KEYPOINT_NAMES
from schemas.skeleton import Keypoint3D

from .pose_2d import Keypoint2D

# ---------------------------------------------------------------------------
# Constantes de MotionBERT Lite
# ---------------------------------------------------------------------------
WINDOW_SIZE = 243   # frames por ventana (fijo en MotionBERT)
STRIDE = 121        # solapamiento 50% entre ventanas

# Número de keypoints esperados por MotionBERT
_NUM_KPTS = 17

# Mapeo de KEYPOINT_NAMES (schemas) a índices de MotionBERT (orden COCO-like)
# MotionBERT usa el orden COCO de 17 puntos
_NAME_TO_MB_IDX: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    # ear -> no en schemas, no mapeado
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Índices de MotionBERT a nombre de keypoint (solo los mapeados)
_MB_IDX_TO_NAME: dict[int, str] = {v: k for k, v in _NAME_TO_MB_IDX.items()}


def _keypoints_to_array(
    kpts: dict[str, Keypoint2D],
    img_w: float = 1920.0,
    img_h: float = 1080.0,
) -> np.ndarray:
    """
    Convierte dict de Keypoint2D a array (17, 3) normalizado a [-1, 1].
    Los puntos no disponibles se colocan en el centro (0, 0).
    """
    arr = np.zeros((_NUM_KPTS, 3), dtype=np.float32)  # (x, y, conf)
    for name, kpt in kpts.items():
        mb_idx = _NAME_TO_MB_IDX.get(name)
        if mb_idx is not None:
            # Normalizar coordenadas a [-1, 1]
            arr[mb_idx, 0] = (kpt.x / img_w) * 2.0 - 1.0
            arr[mb_idx, 1] = (kpt.y / img_h) * 2.0 - 1.0
            arr[mb_idx, 2] = kpt.confidence

    # Para keypoints derivados (neck, mid_hip, mid_shoulder) usar índices vacíos
    # Calcular neck como promedio de hombros si está disponible
    ls_idx, rs_idx = _NAME_TO_MB_IDX.get("left_shoulder"), _NAME_TO_MB_IDX.get("right_shoulder")
    if ls_idx is not None and rs_idx is not None:
        neck_x = (arr[ls_idx, 0] + arr[rs_idx, 0]) / 2.0
        neck_y = (arr[ls_idx, 1] + arr[rs_idx, 1]) / 2.0
        neck_conf = (arr[ls_idx, 2] + arr[rs_idx, 2]) / 2.0
        # Guardar en slot 1 (nose area, pero MotionBERT lo usa como neck)
        # En realidad se usará para derivar neck en la salida

    return arr


def _array_to_keypoints_3d(
    arr_3d: np.ndarray,
) -> dict[str, Keypoint3D]:
    """
    Convierte array (17, 3) de MotionBERT a dict de Keypoint3D.
    Agrega keypoints derivados (neck, mid_hip, mid_shoulder).
    """
    result: dict[str, Keypoint3D] = {}

    for mb_idx, name in _MB_IDX_TO_NAME.items():
        if mb_idx < len(arr_3d):
            x, y, z = arr_3d[mb_idx]
            result[name] = Keypoint3D(
                x=float(x),
                y=float(y),
                z=float(z),
                confidence=0.9,  # MotionBERT no retorna confianza explícita
            )

    # Derivar keypoints adicionales
    _derive_extra_3d(result)

    # Asegurarse de que todos los KEYPOINT_NAMES estén presentes
    for name in KEYPOINT_NAMES:
        if name not in result:
            result[name] = Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0)

    return result


def _derive_extra_3d(kpts: dict[str, Keypoint3D]) -> None:
    """Deriva neck, mid_hip y mid_shoulder por promedio."""
    # neck
    if "left_shoulder" in kpts and "right_shoulder" in kpts:
        ls, rs = kpts["left_shoulder"], kpts["right_shoulder"]
        kpts["neck"] = Keypoint3D(
            x=(ls.x + rs.x) / 2.0,
            y=(ls.y + rs.y) / 2.0,
            z=(ls.z + rs.z) / 2.0,
            confidence=(ls.confidence + rs.confidence) / 2.0,
        )

    # mid_hip
    if "left_hip" in kpts and "right_hip" in kpts:
        lh, rh = kpts["left_hip"], kpts["right_hip"]
        kpts["mid_hip"] = Keypoint3D(
            x=(lh.x + rh.x) / 2.0,
            y=(lh.y + rh.y) / 2.0,
            z=(lh.z + rh.z) / 2.0,
            confidence=(lh.confidence + rh.confidence) / 2.0,
        )

    # mid_shoulder = neck
    if "neck" in kpts:
        n = kpts["neck"]
        kpts["mid_shoulder"] = Keypoint3D(
            x=n.x, y=n.y, z=n.z, confidence=n.confidence
        )


def _pad_sequence(
    seq: list[dict[str, Keypoint2D]], target_len: int
) -> list[dict[str, Keypoint2D]]:
    """
    Hace padding de la secuencia a target_len frames.
    Rellena al inicio con el primer frame y al final con el último.
    """
    if len(seq) >= target_len:
        return seq[:target_len]
    pad_before = (target_len - len(seq)) // 2
    pad_after = target_len - len(seq) - pad_before
    return [seq[0]] * pad_before + list(seq) + [seq[-1]] * pad_after


def _build_input_tensor(
    window: list[dict[str, Keypoint2D]],
    img_w: float = 1920.0,
    img_h: float = 1080.0,
) -> np.ndarray:
    """
    Construye el tensor de entrada (1, T, 17, 3) para MotionBERT.
    """
    T = len(window)
    tensor = np.zeros((1, T, _NUM_KPTS, 3), dtype=np.float32)
    for t, kpts in enumerate(window):
        tensor[0, t] = _keypoints_to_array(kpts, img_w, img_h)
    return tensor


class PoseLifter3D:
    """
    Lifting 2D→3D usando MotionBERT Lite.
    Procesa ventanas de 243 frames de keypoints 2D normalizados.
    """

    WINDOW_SIZE: int = WINDOW_SIZE
    STRIDE: int = STRIDE

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """
        Args:
            model_path: ruta al archivo ONNX de MotionBERT Lite.
            device: 'cpu' o 'cuda'.
        """
        self.model_path = str(model_path)
        self.device = device
        self._session = None
        logger.debug(f"PoseLifter3D inicializado con modelo: {self.model_path}")

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
                logger.debug("ONNX session de lifting 3D cargada")
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo cargar el modelo ONNX '{self.model_path}': {e}"
                ) from e
        return self._session

    def _run_window(
        self, window: list[dict[str, Keypoint2D]]
    ) -> list[dict[str, Keypoint3D]]:
        """
        Ejecuta MotionBERT sobre una ventana de exactamente WINDOW_SIZE frames.

        Returns:
            Lista de WINDOW_SIZE dicts de Keypoint3D.
        """
        session = self._get_session()
        inp = _build_input_tensor(window)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})

        # MotionBERT output: (1, T, 17, 3) en coordenadas root-relative
        output_3d = outputs[0][0]  # (T, 17, 3)

        result = []
        for t in range(output_3d.shape[0]):
            kpts = _array_to_keypoints_3d(output_3d[t])
            result.append(kpts)
        return result

    def lift(
        self,
        keypoints_2d_sequence: list[dict[str, Keypoint2D]],
    ) -> list[dict[str, Keypoint3D]]:
        """
        Eleva una secuencia de keypoints 2D a 3D usando ventana deslizante.

        Args:
            keypoints_2d_sequence: N frames de keypoints 2D.

        Returns:
            N frames de keypoints 3D (root-relative, no escalados a metros).

        Notas:
            - Si N < WINDOW_SIZE, hace padding.
            - Usa ventana deslizante con STRIDE para secuencias largas.
            - Los frames solapados se promedian.
        """
        N = len(keypoints_2d_sequence)
        if N == 0:
            return []

        # Acumuladores para promediar ventanas solapadas
        # Cada frame puede tener contribuciones de múltiples ventanas
        # Usar arrays de acumulación
        accumulated: list[Optional[dict[str, Keypoint3D]]] = [None] * N
        counts: list[int] = [0] * N

        # Construir lista de ventanas (start_frame, end_frame, padded_window)
        windows: list[tuple[int, int, list]] = []

        if N <= self.WINDOW_SIZE:
            # Un solo chunk con padding
            padded = _pad_sequence(keypoints_2d_sequence, self.WINDOW_SIZE)
            # El centro de la ventana paddeada corresponde al rango [0, N)
            pad_before = (self.WINDOW_SIZE - N) // 2
            windows.append((0, N, padded))
        else:
            # Ventana deslizante
            start = 0
            while start < N:
                end = min(start + self.WINDOW_SIZE, N)
                chunk = list(keypoints_2d_sequence[start:end])
                if len(chunk) < self.WINDOW_SIZE:
                    chunk = _pad_sequence(chunk, self.WINDOW_SIZE)
                    # El chunk corto cubre [start, N)
                    windows.append((start, N, chunk))
                else:
                    windows.append((start, end, chunk))
                start += self.STRIDE
                if start >= N:
                    break

        # Procesar cada ventana
        for win_start, win_end, padded in windows:
            win_results = self._run_window(padded)
            # Tomar solo los frames que corresponden al rango [win_start, win_end)
            win_len = win_end - win_start
            pad_before = (self.WINDOW_SIZE - win_len) // 2 if win_len < self.WINDOW_SIZE else 0
            relevant = win_results[pad_before: pad_before + win_len]

            for local_i, frame_kpts in enumerate(relevant):
                global_i = win_start + local_i
                if global_i >= N:
                    break
                if accumulated[global_i] is None:
                    accumulated[global_i] = frame_kpts
                    counts[global_i] = 1
                else:
                    # Promediar keypoints
                    accumulated[global_i] = _average_keypoints(
                        accumulated[global_i], frame_kpts, counts[global_i]
                    )
                    counts[global_i] += 1

        # Rellenar frames no procesados con keypoints vacíos
        final: list[dict[str, Keypoint3D]] = []
        for i in range(N):
            if accumulated[i] is None:
                empty: dict[str, Keypoint3D] = {
                    name: Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0)
                    for name in KEYPOINT_NAMES
                }
                final.append(empty)
            else:
                final.append(accumulated[i])
        return final


def _average_keypoints(
    existing: dict[str, Keypoint3D],
    new: dict[str, Keypoint3D],
    count: int,
) -> dict[str, Keypoint3D]:
    """Promedia dos dicts de Keypoint3D ponderando por count."""
    result = {}
    for name in KEYPOINT_NAMES:
        e = existing.get(name, Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0))
        n = new.get(name, Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.0))
        result[name] = Keypoint3D(
            x=(e.x * count + n.x) / (count + 1),
            y=(e.y * count + n.y) / (count + 1),
            z=(e.z * count + n.z) / (count + 1),
            confidence=(e.confidence * count + n.confidence) / (count + 1),
        )
    return result
