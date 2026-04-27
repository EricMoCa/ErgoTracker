"""
pose_pipeline/height.py
Conversión de coordenadas 3D normalizadas a metros reales usando la altura de la persona.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from schemas.skeleton import Keypoint3D


class HeightAnchor:
    """
    Convierte coordenadas 3D root-relative normalizadas a metros reales
    usando la altura de la persona como ancla de escala.
    """

    def compute_scale_factor(
        self,
        skeleton_3d: dict[str, Keypoint3D],
        person_height_cm: float,
    ) -> float:
        """
        Calcula el factor unidad→metros usando la altura real de la persona.
        Método: distancia desde cabeza (nose) hasta punto medio de tobillos.
        """
        nose = skeleton_3d.get("nose")
        left_ankle = skeleton_3d.get("left_ankle")
        right_ankle = skeleton_3d.get("right_ankle")

        if nose is None:
            logger.warning("Keypoint 'nose' no disponible para cálculo de escala")
            return person_height_cm / 100.0

        if left_ankle is not None and right_ankle is not None:
            ankle_y = (left_ankle.y + right_ankle.y) / 2.0
        elif left_ankle is not None:
            ankle_y = left_ankle.y
        elif right_ankle is not None:
            ankle_y = right_ankle.y
        else:
            logger.warning("Tobillos no disponibles para cálculo de escala")
            return person_height_cm / 100.0

        skeleton_height = abs(nose.y - ankle_y)
        if skeleton_height < 1e-9:
            logger.warning("Altura del esqueleto es 0 — usando altura por defecto")
            return person_height_cm / 100.0

        scale = (person_height_cm / 100.0) / skeleton_height
        return float(scale)

    def estimate_height_from_skeleton(
        self,
        skeleton_3d: dict[str, Keypoint3D],
    ) -> float:
        """
        Fallback: estima la altura en cm usando proporciones antropométricas.
        Relación cabeza/cuerpo estándar: altura = 7.5 * altura_cabeza.
        Retorna estimación en cm.
        """
        nose = skeleton_3d.get("nose")
        neck = skeleton_3d.get("neck")

        if nose is None or neck is None:
            return 170.0  # valor por defecto

        head_height = abs(nose.y - neck.y)
        if head_height < 1e-9:
            return 170.0

        estimated_m = head_height * 7.5
        return float(estimated_m * 100.0)

    def apply_scale(
        self,
        skeletons: list[dict[str, Keypoint3D]],
        scale_m_per_unit: float,
    ) -> list[dict[str, Keypoint3D]]:
        """
        Aplica el factor de escala a todos los keypoints.
        Retorna nuevas instancias con coordenadas en metros.
        """
        result = []
        for skeleton in skeletons:
            scaled = {}
            for name, kp in skeleton.items():
                scaled[name] = Keypoint3D(
                    x=kp.x * scale_m_per_unit,
                    y=kp.y * scale_m_per_unit,
                    z=kp.z * scale_m_per_unit,
                    confidence=kp.confidence,
                    occluded=kp.occluded,
                )
            result.append(scaled)
        return result
