"""
reports/video_annotator.py
Genera un video side-by-side:
  Panel izquierdo : video original con esqueleto 3D superpuesto (proyección ortográfica)
  Panel derecho   : vista 3D de mundo desde un ángulo isométrico fijo

Inspirado en la salida visual de WHAM.  Todo en OpenCV + numpy (sin matplotlib),
por lo que la velocidad de render es aceptable incluso en CPU.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from schemas import SkeletonSequence, FrameErgonomicScore, RiskLevel, Skeleton3D

# ---------------------------------------------------------------------------
# Paleta de colores por riesgo (BGR para OpenCV)
# ---------------------------------------------------------------------------
_RISK_COLOR: dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.NEGLIGIBLE: (94,  197,  34),
    RiskLevel.LOW:        (86,  239, 134),
    RiskLevel.MEDIUM:     (50,  205, 253),
    RiskLevel.HIGH:       (22,  115, 249),
    RiskLevel.VERY_HIGH:  (68,   68, 239),
}
_WHITE  = (240, 240, 240)
_GRAY   = (120, 120, 120)
_DARK   = ( 20,  20,  30)

_RISK_LABEL = {
    RiskLevel.NEGLIGIBLE: "Negligible",
    RiskLevel.LOW:        "Bajo",
    RiskLevel.MEDIUM:     "Medio",
    RiskLevel.HIGH:       "ALTO",
    RiskLevel.VERY_HIGH:  "MUY ALTO",
}

# ---------------------------------------------------------------------------
# Conexiones del esqueleto
# ---------------------------------------------------------------------------
_CONNECTIONS: list[tuple[str, str]] = [
    ("nose",           "neck"),
    ("neck",           "mid_shoulder"),
    ("mid_shoulder",   "right_shoulder"),
    ("mid_shoulder",   "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("neck",           "mid_hip"),
    ("mid_hip",        "right_hip"),
    ("mid_hip",        "left_hip"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_eye",      "nose"),
    ("left_eye",       "nose"),
]


# ---------------------------------------------------------------------------
# VideoAnnotator
# ---------------------------------------------------------------------------
class VideoAnnotator:
    """
    Genera un video MP4 side-by-side con visualización ergonómica:
      Izquierda → video original + esqueleto superpuesto
      Derecha   → vista 3D de mundo (proyección isométrica en OpenCV)
    """

    # Ángulos de la vista isométrica (grados)
    _AZIM_DEG = -50.0
    _ELEV_DEG =  20.0

    def generate(
        self,
        video_path: str,
        skeleton_seq: SkeletonSequence,
        frame_scores: list[FrameErgonomicScore],
        output_path: str,
    ) -> str:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python requerido: pip install opencv-python")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W_out  = W * 2  # ambos paneles del mismo ancho

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W_out, H))

        # Índices rápidos
        skel_idx  = {s.frame_idx: s for s in skeleton_seq.frames}
        score_idx = {fs.frame_idx: fs for fs in frame_scores}
        sorted_keys = sorted(skel_idx.keys())

        # Precomputar matriz de rotación para la vista 3D
        R = self._build_rotation(self._AZIM_DEG, self._ELEV_DEG)

        cur_skel:  Optional[Skeleton3D]             = None
        cur_score: Optional[FrameErgonomicScore]    = None
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            nearest = self._nearest(frame_no, sorted_keys)
            if nearest is not None:
                cur_skel  = skel_idx.get(nearest)
                cur_score = score_idx.get(nearest)

            risk   = cur_score.overall_risk if cur_score else RiskLevel.NEGLIGIBLE
            color  = _RISK_COLOR.get(risk, _WHITE)

            # Panel izquierdo: video con overlay
            left = self._draw_video_overlay(frame.copy(), cur_skel, cur_score, color, W, H)

            # Panel derecho: vista 3D de mundo
            right = self._draw_world_3d(cur_skel, cur_score, color, R, W, H)

            combined = np.hstack([left, right])
            writer.write(combined)
            frame_no += 1

        cap.release()
        writer.release()

        size_kb = Path(output_path).stat().st_size // 1024
        logger.success(f"Video side-by-side generado: {output_path} ({size_kb} KB)")
        return str(output_path)

    # ------------------------------------------------------------------
    # Panel izquierdo: overlay sobre el video original
    # ------------------------------------------------------------------
    def _draw_video_overlay(self, frame, skeleton, score, color, W, H):
        import cv2

        if skeleton is None:
            return frame

        scale, ox, oy = self._video_projection_params(skeleton, W, H)

        def proj(name: str):
            kp = skeleton.keypoints.get(name)
            if kp is None or kp.confidence < 0.1:
                return None
            return (int(ox + kp.x * scale), int(oy - kp.y * scale))

        # HUD de riesgo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (230, 56), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        risk_label = _RISK_LABEL.get(score.overall_risk if score else RiskLevel.NEGLIGIBLE, "")
        cv2.putText(frame, f"Riesgo: {risk_label}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        if score and score.reba:
            cv2.putText(frame, f"REBA {score.reba.total}/15", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1, cv2.LINE_AA)

        # Etiqueta panel
        cv2.putText(frame, "Video + Esqueleto", (W - 170, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1, cv2.LINE_AA)

        # Huesos
        for a_n, b_n in _CONNECTIONS:
            pa, pb = proj(a_n), proj(b_n)
            if pa and pb:
                cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)

        # Articulaciones
        for name, kp in skeleton.keypoints.items():
            pt = proj(name)
            if pt is None:
                continue
            r = 6 if name in ("mid_hip", "neck", "mid_shoulder") else 4
            cv2.circle(frame, pt, r + 2, _DARK,  -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r,     color,  -1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Panel derecho: vista 3D de mundo
    # ------------------------------------------------------------------
    def _draw_world_3d(self, skeleton, score, color, R, W, H):
        import cv2

        canvas = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)

        # Cuadrícula del suelo
        self._draw_floor_grid(canvas, R, W, H)

        # Etiqueta panel
        cv2.putText(canvas, "Vista 3D Mundo", (W - 155, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1, cv2.LINE_AA)

        # Ángulo de vista
        cv2.putText(canvas, f"Az={self._AZIM_DEG:.0f}°  El={self._ELEV_DEG:.0f}°",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1, cv2.LINE_AA)

        if skeleton is None:
            return canvas

        scale, cx, cy = self._world_projection_params(skeleton, W, H)

        def proj3d(name: str):
            kp = skeleton.keypoints.get(name)
            if kp is None or kp.confidence < 0.1:
                return None
            v = np.array([kp.x, kp.y, kp.z])
            r = R @ v
            return (int(cx + r[0] * scale), int(cy - r[1] * scale))

        # Huesos
        for a_n, b_n in _CONNECTIONS:
            pa, pb = proj3d(a_n), proj3d(b_n)
            if pa and pb:
                cv2.line(canvas, pa, pb, color, 3, cv2.LINE_AA)

        # Articulaciones
        for name, kp in skeleton.keypoints.items():
            pt = proj3d(name)
            if pt is None:
                continue
            r = 7 if name in ("mid_hip", "neck", "mid_shoulder") else 5
            cv2.circle(canvas, pt, r + 2, _DARK,  -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r,     color,  -1, cv2.LINE_AA)

        # HUD de score en 3D
        risk_label = _RISK_LABEL.get(score.overall_risk if score else RiskLevel.NEGLIGIBLE, "")
        cv2.putText(canvas, f"Riesgo: {risk_label}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        if score and score.reba:
            cv2.putText(canvas, f"REBA {score.reba.total}/15  OWAS {score.owas.action_level if score.owas else '-'}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1, cv2.LINE_AA)

        return canvas

    def _draw_floor_grid(self, canvas, R, W, H):
        import cv2
        _, cx, cy = self._grid_projection_params(W, H)
        scale = (H * 0.25)

        grid_range = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for gx in grid_range:
            # Línea a lo largo de z
            for gz in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                v1 = R @ np.array([gx,  0.0, -1.0])
                v2 = R @ np.array([gx,  0.0,  1.0])
                p1 = (int(cx + v1[0] * scale), int(cy - v1[1] * scale))
                p2 = (int(cx + v2[0] * scale), int(cy - v2[1] * scale))
                cv2.line(canvas, p1, p2, (40, 40, 55), 1, cv2.LINE_AA)
            v1 = R @ np.array([-1.0, 0.0, gx])
            v2 = R @ np.array([ 1.0, 0.0, gx])
            p1 = (int(cx + v1[0] * scale), int(cy - v1[1] * scale))
            p2 = (int(cx + v2[0] * scale), int(cy - v2[1] * scale))
            cv2.line(canvas, p1, p2, (40, 40, 55), 1, cv2.LINE_AA)

        # Ejes X (rojo), Y (verde), Z (azul)
        for axis, col in [
            (np.array([0.5, 0, 0]),   (60, 60, 200)),
            (np.array([0, 0.5, 0]),   (60, 200,  60)),
            (np.array([0, 0, 0.5]),   (200,  60,  60)),
        ]:
            v = R @ axis
            pt = (int(cx + v[0] * scale), int(cy - v[1] * scale))
            org = (int(cx), int(cy))
            cv2.line(canvas, org, pt, col, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Parámetros de proyección
    # ------------------------------------------------------------------
    def _video_projection_params(self, skeleton: Skeleton3D, W: int, H: int):
        h_m = skeleton.person_height_cm / 100.0
        scale = (H * 0.75) / h_m
        ox = W // 2
        oy = int(H * 0.68)
        return scale, ox, oy

    def _world_projection_params(self, skeleton: Skeleton3D, W: int, H: int):
        h_m = skeleton.person_height_cm / 100.0
        scale = (H * 0.60) / h_m
        cx = W // 2
        cy = int(H * 0.60)
        return scale, cx, cy

    def _grid_projection_params(self, W: int, H: int):
        return None, W // 2, int(H * 0.78)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    @staticmethod
    def _build_rotation(azim_deg: float, elev_deg: float) -> np.ndarray:
        azim = math.radians(azim_deg)
        elev = math.radians(elev_deg)
        Ry = np.array([
            [ math.cos(azim), 0, math.sin(azim)],
            [ 0,              1, 0             ],
            [-math.sin(azim), 0, math.cos(azim)],
        ])
        Rx = np.array([
            [1, 0,              0             ],
            [0, math.cos(elev), -math.sin(elev)],
            [0, math.sin(elev),  math.cos(elev)],
        ])
        return Rx @ Ry

    @staticmethod
    def _nearest(idx: int, sorted_idxs: list[int]) -> Optional[int]:
        if not sorted_idxs:
            return None
        return min(sorted_idxs, key=lambda x: abs(x - idx))
