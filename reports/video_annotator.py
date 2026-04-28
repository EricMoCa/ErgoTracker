"""
reports/video_annotator.py
Genera un video side-by-side:
  Panel izquierdo : video original con esqueleto 3D superpuesto (proyección ortográfica)
  Panel derecho   : vista 3D de mundo estilo WHAM (segmentos gruesos, trail de cadera)

Todo en OpenCV + numpy (sin matplotlib).
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
_CYAN   = (255, 242,   0)  # BGR for cyan-ish accent

_RISK_LABEL = {
    RiskLevel.NEGLIGIBLE: "Negligible",
    RiskLevel.LOW:        "Bajo",
    RiskLevel.MEDIUM:     "Medio",
    RiskLevel.HIGH:       "ALTO",
    RiskLevel.VERY_HIGH:  "MUY ALTO",
}

# ---------------------------------------------------------------------------
# Conexiones del esqueleto — (nombre_a, nombre_b, es_tronco)
# ---------------------------------------------------------------------------
_CONNECTIONS: list[tuple[str, str, bool]] = [
    ("nose",           "neck",           False),
    ("neck",           "mid_shoulder",   True),
    ("mid_shoulder",   "right_shoulder", True),
    ("mid_shoulder",   "left_shoulder",  True),
    ("right_shoulder", "right_elbow",    False),
    ("right_elbow",    "right_wrist",    False),
    ("left_shoulder",  "left_elbow",     False),
    ("left_elbow",     "left_wrist",     False),
    ("neck",           "mid_hip",        True),
    ("mid_hip",        "right_hip",      True),
    ("mid_hip",        "left_hip",       True),
    ("right_hip",      "right_knee",     True),
    ("right_knee",     "right_ankle",    False),
    ("left_hip",       "left_knee",      True),
    ("left_knee",      "left_ankle",     False),
    ("right_eye",      "nose",           False),
    ("left_eye",       "nose",           False),
]

_TRAIL_LEN = 60  # frames de trail de cadera


# ---------------------------------------------------------------------------
# VideoAnnotator
# ---------------------------------------------------------------------------
class VideoAnnotator:
    """
    Genera un video MP4 side-by-side con visualización ergonómica:
      Izquierda → video original + esqueleto superpuesto (alineado al cuerpo)
      Derecha   → vista 3D estilo WHAM (segmentos gruesos, cabeza, torso, trail)
    """

    # Ángulos de la vista isométrica — más perspectiva que antes
    _AZIM_DEG = -40.0
    _ELEV_DEG =  28.0

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
        W_out  = W * 2

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W_out, H))

        skel_idx  = {s.frame_idx: s for s in skeleton_seq.frames}
        score_idx = {fs.frame_idx: fs for fs in frame_scores}
        sorted_keys = sorted(skel_idx.keys())

        R = self._build_rotation(self._AZIM_DEG, self._ELEV_DEG)

        cur_skel:  Optional[Skeleton3D]          = None
        cur_score: Optional[FrameErgonomicScore] = None
        frame_no = 0
        hip_trail: list[tuple[float, float, float]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            nearest = self._nearest(frame_no, sorted_keys)
            if nearest is not None:
                cur_skel  = skel_idx.get(nearest)
                cur_score = score_idx.get(nearest)

            # Acumular trail de cadera
            if cur_skel:
                hip = (cur_skel.keypoints.get("mid_hip")
                       or cur_skel.keypoints.get("left_hip"))
                if hip and hip.confidence > 0.1:
                    hip_trail.append((hip.x, hip.y, hip.z))
                    hip_trail = hip_trail[-_TRAIL_LEN:]

            risk  = cur_score.overall_risk if cur_score else RiskLevel.NEGLIGIBLE
            color = _RISK_COLOR.get(risk, _WHITE)

            left  = self._draw_video_overlay(frame.copy(), cur_skel, cur_score, color, W, H)
            right = self._draw_world_3d(cur_skel, cur_score, color, R, W, H, hip_trail)

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
            if kp is None or kp.confidence < 0.15:
                return None
            return (int(ox + kp.x * scale), int(oy - kp.y * scale))

        # Bounding box semitransparente del cuerpo para orientación
        pts = [proj(n) for n in skeleton.keypoints if proj(n) is not None]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 12
            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x2, y2 = min(W, max(xs) + pad), min(H, max(ys) + pad)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # HUD de riesgo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (240, 60), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        risk_label = _RISK_LABEL.get(score.overall_risk if score else RiskLevel.NEGLIGIBLE, "")
        cv2.putText(frame, f"Riesgo: {risk_label}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        if score and score.reba:
            cv2.putText(frame, f"REBA {score.reba.total}/15", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, "Video + Esqueleto", (W - 175, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1, cv2.LINE_AA)

        # Huesos — tronco 5px, extremidades 3px
        for a_n, b_n, is_trunk in _CONNECTIONS:
            pa, pb = proj(a_n), proj(b_n)
            if pa and pb:
                thickness = 5 if is_trunk else 3
                cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

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
    # Panel derecho: vista 3D estilo WHAM
    # ------------------------------------------------------------------
    def _draw_world_3d(self, skeleton, score, color, R, W, H,
                       hip_trail: list[tuple[float, float, float]]):
        import cv2

        # Fondo con gradiente vertical (negro arriba → azul oscuro abajo)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for row in range(H):
            t = row / max(H - 1, 1)
            b = int(18 + t * 10)
            g = int(18 + t * 8)
            r = int(28 + t * 18)
            canvas[row, :] = (b, g, r)

        # Cuadrícula del suelo
        self._draw_floor_grid(canvas, R, W, H)

        # Etiqueta y ángulos
        cv2.putText(canvas, "Vista 3D Mundo", (W - 160, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Az={self._AZIM_DEG:.0f}°  El={self._ELEV_DEG:.0f}°",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1, cv2.LINE_AA)

        scale, cx, cy = self._world_projection_params(skeleton, W, H) if skeleton else (
            H * 0.55 / 1.7, W // 2, int(H * 0.62)
        )

        def proj3d(x, y, z):
            v = R @ np.array([x, y, z])
            return (int(cx + v[0] * scale), int(cy - v[1] * scale))

        def proj_kp(name: str):
            if skeleton is None:
                return None
            kp = skeleton.keypoints.get(name)
            if kp is None or kp.confidence < 0.1:
                return None
            return proj3d(kp.x, kp.y, kp.z)

        # Trail de trayectoria de cadera (puntos con alpha decreciente)
        if hip_trail:
            for i, (hx, hy, hz) in enumerate(hip_trail):
                t = i / max(len(hip_trail) - 1, 1)
                alpha = int(30 + t * 160)
                pt = proj3d(hx, hy, hz)
                r_dot = max(1, int(t * 4))
                # Dibujar círculo translúcido como punto del trail
                overlay = canvas.copy()
                cv2.circle(overlay, pt, r_dot, (255, 200, 50), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha / 255.0, canvas, 1 - alpha / 255.0, 0, canvas)

        if skeleton is None:
            return canvas

        # Sombra de pies: elipse aplastada bajo tobillos
        ankle_pts = []
        for side in ("left_ankle", "right_ankle"):
            kp = skeleton.keypoints.get(side)
            if kp and kp.confidence > 0.1:
                ankle_pts.append(proj3d(kp.x, 0.0, kp.z))
        if ankle_pts:
            shadow_cx = sum(p[0] for p in ankle_pts) // len(ankle_pts)
            shadow_cy = sum(p[1] for p in ankle_pts) // len(ankle_pts)
            shadow_canvas = canvas.copy()
            cv2.ellipse(shadow_canvas, (shadow_cx, shadow_cy + 4),
                        (28, 8), 0, 0, 360, (50, 50, 50), -1, cv2.LINE_AA)
            cv2.addWeighted(shadow_canvas, 0.5, canvas, 0.5, 0, canvas)

        # Torso: elipse semi-transparente entre mid_shoulder y mid_hip
        p_sh = proj_kp("mid_shoulder")
        p_hp = proj_kp("mid_hip")
        if p_sh and p_hp:
            torso_cx = (p_sh[0] + p_hp[0]) // 2
            torso_cy = (p_sh[1] + p_hp[1]) // 2
            torso_h  = max(8, abs(p_sh[1] - p_hp[1]) // 2)
            torso_w  = max(6, torso_h // 2)
            torso_overlay = canvas.copy()
            cv2.ellipse(torso_overlay, (torso_cx, torso_cy),
                        (torso_w, torso_h), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.addWeighted(torso_overlay, 0.25, canvas, 0.75, 0, canvas)

        # Huesos — tronco 16px, muslos 14px, brazos/piernas 10px
        TRUNK_JOINTS = {"neck", "mid_shoulder", "mid_hip", "left_hip", "right_hip",
                        "left_shoulder", "right_shoulder"}
        THIGH_PAIRS  = {("right_hip","right_knee"), ("left_hip","left_knee")}

        for a_n, b_n, is_trunk in _CONNECTIONS:
            pa, pb = proj_kp(a_n), proj_kp(b_n)
            if pa and pb:
                if is_trunk and a_n in TRUNK_JOINTS and b_n in TRUNK_JOINTS:
                    if (a_n, b_n) in THIGH_PAIRS or (b_n, a_n) in THIGH_PAIRS:
                        thickness = 14
                    else:
                        thickness = 16
                else:
                    thickness = 10
                cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)

        # Cabeza: círculo relleno centrado en nose
        p_nose = proj_kp("nose")
        p_neck = proj_kp("neck")
        if p_nose:
            head_r = 14
            cv2.circle(canvas, p_nose, head_r + 2, _DARK,  -1, cv2.LINE_AA)
            cv2.circle(canvas, p_nose, head_r,     color,  -1, cv2.LINE_AA)
        elif p_neck:
            cv2.circle(canvas, p_neck, 10, color, -1, cv2.LINE_AA)

        # Articulaciones secundarias
        for name, kp in skeleton.keypoints.items():
            if name in ("nose", "neck", "mid_shoulder", "mid_hip"):
                continue
            pt = proj_kp(name)
            if pt is None:
                continue
            r = 7 if name in ("left_hip", "right_hip", "left_knee", "right_knee") else 5
            cv2.circle(canvas, pt, r + 2, _DARK, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r,    color,  -1, cv2.LINE_AA)

        # HUD de score
        risk_label = _RISK_LABEL.get(score.overall_risk if score else RiskLevel.NEGLIGIBLE, "")
        cv2.putText(canvas, f"Riesgo: {risk_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        if score and score.reba:
            owas_str = f"  OWAS {score.owas.action_level}" if score.owas else ""
            cv2.putText(canvas, f"REBA {score.reba.total}/15{owas_str}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1, cv2.LINE_AA)

        return canvas

    def _draw_floor_grid(self, canvas, R, W, H):
        import cv2
        cx, cy = W // 2, int(H * 0.78)
        scale  = H * 0.25

        grid_range = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for gx in grid_range:
            v1 = R @ np.array([gx, 0.0, -1.0])
            v2 = R @ np.array([gx, 0.0,  1.0])
            p1 = (int(cx + v1[0] * scale), int(cy - v1[1] * scale))
            p2 = (int(cx + v2[0] * scale), int(cy - v2[1] * scale))
            cv2.line(canvas, p1, p2, (40, 40, 55), 1, cv2.LINE_AA)
        for gz in grid_range:
            v1 = R @ np.array([-1.0, 0.0, gz])
            v2 = R @ np.array([ 1.0, 0.0, gz])
            p1 = (int(cx + v1[0] * scale), int(cy - v1[1] * scale))
            p2 = (int(cx + v2[0] * scale), int(cy - v2[1] * scale))
            cv2.line(canvas, p1, p2, (40, 40, 55), 1, cv2.LINE_AA)

        # Ejes X (rojo-BGR), Y (verde-BGR), Z (azul-BGR)
        for axis, col in [
            (np.array([0.5, 0, 0]),   (60, 60, 200)),
            (np.array([0, 0.5, 0]),   (60, 200,  60)),
            (np.array([0, 0, 0.5]),   (200,  60,  60)),
        ]:
            v  = R @ axis
            pt = (int(cx + v[0] * scale), int(cy - v[1] * scale))
            cv2.line(canvas, (cx, cy), pt, col, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Parámetros de proyección
    # ------------------------------------------------------------------
    def _video_projection_params(self, skeleton: Skeleton3D, W: int, H: int):
        """Centra el esqueleto sobre el cuerpo real en el panel de video."""
        h_m = skeleton.person_height_cm / 100.0
        scale = H * 0.82 / h_m

        # Centrar horizontalmente sobre la cadera (no en W//2 fijo)
        hip = skeleton.keypoints.get("mid_hip") or skeleton.keypoints.get("left_hip")
        hip_x_m = (hip.x if hip and hip.confidence > 0.1 else 0.0)
        ox = int(W / 2 - hip_x_m * scale)

        # Alinear verticalmente desde los tobillos
        ankle_ys = [
            k.y for n in ("left_ankle", "right_ankle")
            if (k := skeleton.keypoints.get(n)) and k.confidence > 0.1
        ]
        avg_ankle_y = (sum(ankle_ys) / len(ankle_ys)) if ankle_ys else 0.0
        oy = int(H * 0.88 - avg_ankle_y * scale)

        return scale, ox, oy

    def _world_projection_params(self, skeleton, W: int, H: int):
        h_m = skeleton.person_height_cm / 100.0
        scale = (H * 0.55) / h_m
        cx = W // 2
        cy = int(H * 0.62)
        return scale, cx, cy

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
            [1, 0,              0              ],
            [0, math.cos(elev), -math.sin(elev)],
            [0, math.sin(elev),  math.cos(elev)],
        ])
        return Rx @ Ry

    @staticmethod
    def _nearest(idx: int, sorted_idxs: list[int]) -> Optional[int]:
        if not sorted_idxs:
            return None
        return min(sorted_idxs, key=lambda x: abs(x - idx))
