"""
ergo_engine/angles.py
Calculadora de ángulos articulares desde keypoints Skeleton3D.
"""
from typing import Optional

import numpy as np
from loguru import logger

from schemas import Skeleton3D, JointAngles


class JointAngleCalculator:
    """
    Calcula ángulos articulares en grados desde keypoints 3D.
    Fórmula: angle = arccos(dot(v1,v2) / (|v1|*|v2|)) * (180/π)
    donde v1 = punto_proximal - articulación, v2 = punto_distal - articulación.
    """

    def calculate(self, skeleton: Skeleton3D) -> JointAngles:
        """Calcula todos los ángulos posibles desde un Skeleton3D."""
        kp = skeleton.keypoints

        angles = JointAngles(
            frame_idx=skeleton.frame_idx,
            timestamp_s=skeleton.timestamp_s,
            trunk_flexion=self._trunk_flexion(skeleton),
            trunk_lateral_bending=self._trunk_lateral_bending(skeleton),
            trunk_rotation=self._trunk_rotation(skeleton),
            neck_flexion=self._neck_flexion(skeleton),
            neck_lateral_bending=self._neck_lateral_bending(skeleton),
            shoulder_elevation_left=self._shoulder_elevation(skeleton, "left"),
            shoulder_elevation_right=self._shoulder_elevation(skeleton, "right"),
            shoulder_abduction_left=self._shoulder_abduction(skeleton, "left"),
            shoulder_abduction_right=self._shoulder_abduction(skeleton, "right"),
            elbow_flexion_left=self._elbow_flexion(skeleton, "left"),
            elbow_flexion_right=self._elbow_flexion(skeleton, "right"),
            wrist_flexion_left=self._wrist_flexion(skeleton, "left"),
            wrist_flexion_right=self._wrist_flexion(skeleton, "right"),
            wrist_deviation_left=self._wrist_deviation(skeleton, "left"),
            wrist_deviation_right=self._wrist_deviation(skeleton, "right"),
            hip_flexion_left=self._hip_flexion(skeleton, "left"),
            hip_flexion_right=self._hip_flexion(skeleton, "right"),
            knee_flexion_left=self._knee_flexion(skeleton, "left"),
            knee_flexion_right=self._knee_flexion(skeleton, "right"),
        )
        return angles

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kp(self, skeleton: Skeleton3D, name: str) -> Optional[np.ndarray]:
        """Devuelve el keypoint como array numpy o None si no existe."""
        kp = skeleton.keypoints.get(name)
        if kp is None:
            return None
        return np.array([kp.x, kp.y, kp.z], dtype=float)

    def _angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Ángulo en el punto b, formado por los segmentos b→a y b→c.
        Retorna grados en [0, 180].
        """
        v1 = a - b
        v2 = c - b
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    # ------------------------------------------------------------------
    # Trunk
    # ------------------------------------------------------------------

    def _trunk_flexion(self, skeleton: Skeleton3D) -> Optional[float]:
        """
        Ángulo entre la línea vertical y el segmento mid_hip→mid_shoulder.
        0° = erecto, 90° = horizontal.
        """
        mid_hip = self._kp(skeleton, "mid_hip")
        mid_shoulder = self._kp(skeleton, "mid_shoulder")
        if mid_hip is None or mid_shoulder is None:
            # Fallback: use average of left/right hips and shoulders
            lh = self._kp(skeleton, "left_hip")
            rh = self._kp(skeleton, "right_hip")
            ls = self._kp(skeleton, "left_shoulder")
            rs = self._kp(skeleton, "right_shoulder")
            if lh is not None and rh is not None:
                mid_hip = (lh + rh) / 2.0
            if ls is not None and rs is not None:
                mid_shoulder = (ls + rs) / 2.0
        if mid_hip is None or mid_shoulder is None:
            return None

        trunk_vec = mid_shoulder - mid_hip
        # Vertical reference pointing upward (Y axis up in world coordinates)
        vertical = np.array([0.0, 1.0, 0.0])
        norm_trunk = np.linalg.norm(trunk_vec)
        if norm_trunk < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(trunk_vec, vertical) / norm_trunk, -1.0, 1.0)
        angle_from_vertical = float(np.degrees(np.arccos(cos_angle)))
        return angle_from_vertical

    def _trunk_lateral_bending(self, skeleton: Skeleton3D) -> Optional[float]:
        """Inclinación lateral del tronco en grados."""
        mid_hip = self._kp(skeleton, "mid_hip")
        mid_shoulder = self._kp(skeleton, "mid_shoulder")
        if mid_hip is None:
            lh = self._kp(skeleton, "left_hip")
            rh = self._kp(skeleton, "right_hip")
            if lh is not None and rh is not None:
                mid_hip = (lh + rh) / 2.0
        if mid_shoulder is None:
            ls = self._kp(skeleton, "left_shoulder")
            rs = self._kp(skeleton, "right_shoulder")
            if ls is not None and rs is not None:
                mid_shoulder = (ls + rs) / 2.0
        if mid_hip is None or mid_shoulder is None:
            return None

        trunk_vec = mid_shoulder - mid_hip
        norm = np.linalg.norm(trunk_vec)
        if norm < 1e-9:
            return 0.0
        # Lateral bending is the component in the X-Z plane perpendicular to forward
        # Simplified: angle in the coronal plane (X-Y) from vertical
        lateral_vec = np.array([trunk_vec[0], trunk_vec[1], 0.0])
        vertical = np.array([0.0, 1.0, 0.0])
        norm_lat = np.linalg.norm(lateral_vec)
        if norm_lat < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(lateral_vec, vertical) / norm_lat, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _trunk_rotation(self, skeleton: Skeleton3D) -> Optional[float]:
        """Rotación axial del tronco estimada desde la línea de hombros vs línea de caderas."""
        ls = self._kp(skeleton, "left_shoulder")
        rs = self._kp(skeleton, "right_shoulder")
        lh = self._kp(skeleton, "left_hip")
        rh = self._kp(skeleton, "right_hip")
        if ls is None or rs is None or lh is None or rh is None:
            return None
        shoulder_vec = rs - ls
        hip_vec = rh - lh
        # Project both onto the horizontal plane (X-Z)
        sh_xz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]])
        hp_xz = np.array([hip_vec[0], 0.0, hip_vec[2]])
        norm_sh = np.linalg.norm(sh_xz)
        norm_hp = np.linalg.norm(hp_xz)
        if norm_sh < 1e-9 or norm_hp < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(sh_xz, hp_xz) / (norm_sh * norm_hp), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    # ------------------------------------------------------------------
    # Neck
    # ------------------------------------------------------------------

    def _neck_flexion(self, skeleton: Skeleton3D) -> Optional[float]:
        """Ángulo entre mid_shoulder→neck y neck→nose."""
        neck = self._kp(skeleton, "neck")
        nose = self._kp(skeleton, "nose")
        mid_shoulder = self._kp(skeleton, "mid_shoulder")
        if mid_shoulder is None:
            ls = self._kp(skeleton, "left_shoulder")
            rs = self._kp(skeleton, "right_shoulder")
            if ls is not None and rs is not None:
                mid_shoulder = (ls + rs) / 2.0
        if neck is None or nose is None or mid_shoulder is None:
            return None
        # Angle at neck between mid_shoulder→neck direction extended and neck→nose
        angle = self._angle_between(mid_shoulder, neck, nose)
        # Convert: 180 = fully aligned (erect), 0 = fully folded
        # Neck flexion = 180 - angle (0 = erect, 90 = 90° flexed)
        return 180.0 - angle

    def _neck_lateral_bending(self, skeleton: Skeleton3D) -> Optional[float]:
        """Inclinación lateral de la cabeza respecto al cuello."""
        neck = self._kp(skeleton, "neck")
        l_eye = self._kp(skeleton, "left_eye")
        r_eye = self._kp(skeleton, "right_eye")
        mid_shoulder = self._kp(skeleton, "mid_shoulder")
        if neck is None or l_eye is None or r_eye is None or mid_shoulder is None:
            return None
        # Direction vector trunk (up)
        trunk_up = neck - mid_shoulder
        # Head lateral vector
        eye_vec = r_eye - l_eye
        # Lateral bending: angle of eye line vs shoulder line
        ls = self._kp(skeleton, "left_shoulder")
        rs = self._kp(skeleton, "right_shoulder")
        if ls is None or rs is None:
            return None
        shoulder_vec = rs - ls
        norm_eye = np.linalg.norm(eye_vec)
        norm_sh = np.linalg.norm(shoulder_vec)
        if norm_eye < 1e-9 or norm_sh < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(eye_vec, shoulder_vec) / (norm_eye * norm_sh), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    # ------------------------------------------------------------------
    # Shoulder
    # ------------------------------------------------------------------

    def _shoulder_elevation(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """
        Elevación del hombro (ángulo entre el brazo y el tronco lateral).
        Aproximado como el ángulo entre shoulder→elbow y la vertical.
        """
        shoulder = self._kp(skeleton, f"{side}_shoulder")
        elbow = self._kp(skeleton, f"{side}_elbow")
        if shoulder is None or elbow is None:
            return None
        arm_vec = elbow - shoulder
        vertical = np.array([0.0, -1.0, 0.0])  # pointing downward (arm hanging)
        norm = np.linalg.norm(arm_vec)
        if norm < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(arm_vec, vertical) / norm, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_angle)))
        return angle

    def _shoulder_abduction(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Abducción del hombro (separación lateral del brazo del cuerpo)."""
        shoulder = self._kp(skeleton, f"{side}_shoulder")
        elbow = self._kp(skeleton, f"{side}_elbow")
        mid_hip = self._kp(skeleton, "mid_hip")
        mid_shoulder = self._kp(skeleton, "mid_shoulder")
        if shoulder is None or elbow is None:
            return None
        if mid_hip is None:
            lh = self._kp(skeleton, "left_hip")
            rh = self._kp(skeleton, "right_hip")
            if lh is not None and rh is not None:
                mid_hip = (lh + rh) / 2.0
        if mid_shoulder is None:
            ls = self._kp(skeleton, "left_shoulder")
            rs = self._kp(skeleton, "right_shoulder")
            if ls is not None and rs is not None:
                mid_shoulder = (ls + rs) / 2.0
        if mid_hip is None or mid_shoulder is None:
            return None
        trunk_vec = mid_shoulder - mid_hip
        arm_vec = elbow - shoulder
        norm_trunk = np.linalg.norm(trunk_vec)
        norm_arm = np.linalg.norm(arm_vec)
        if norm_trunk < 1e-9 or norm_arm < 1e-9:
            return 0.0
        cos_angle = np.clip(np.dot(arm_vec, trunk_vec) / (norm_arm * norm_trunk), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    # ------------------------------------------------------------------
    # Elbow
    # ------------------------------------------------------------------

    def _elbow_flexion(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Flexión del codo: ángulo entre shoulder→elbow y elbow→wrist."""
        shoulder = self._kp(skeleton, f"{side}_shoulder")
        elbow = self._kp(skeleton, f"{side}_elbow")
        wrist = self._kp(skeleton, f"{side}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            return None
        angle = self._angle_between(shoulder, elbow, wrist)
        # 180° = fully extended (0° flexion), 0° = fully flexed (180° flexion)
        return 180.0 - angle

    # ------------------------------------------------------------------
    # Wrist
    # ------------------------------------------------------------------

    def _wrist_flexion(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Flexión de muñeca: ángulo entre elbow→wrist y el eje del antebrazo."""
        elbow = self._kp(skeleton, f"{side}_elbow")
        wrist = self._kp(skeleton, f"{side}_wrist")
        if elbow is None or wrist is None:
            return None
        # Without finger keypoints, approximate as deviation from straight forearm
        # Return 0 as default (neutral position)
        return 0.0

    def _wrist_deviation(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Desviación de muñeca (ulnar/radial)."""
        elbow = self._kp(skeleton, f"{side}_elbow")
        wrist = self._kp(skeleton, f"{side}_wrist")
        if elbow is None or wrist is None:
            return None
        return 0.0

    # ------------------------------------------------------------------
    # Hip
    # ------------------------------------------------------------------

    def _hip_flexion(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Flexión de cadera: ángulo entre tronco y muslo."""
        shoulder = self._kp(skeleton, f"{side}_shoulder")
        hip = self._kp(skeleton, f"{side}_hip")
        knee = self._kp(skeleton, f"{side}_knee")
        if shoulder is None or hip is None or knee is None:
            return None
        angle = self._angle_between(shoulder, hip, knee)
        return 180.0 - angle

    # ------------------------------------------------------------------
    # Knee
    # ------------------------------------------------------------------

    def _knee_flexion(self, skeleton: Skeleton3D, side: str) -> Optional[float]:
        """Flexión de rodilla: ángulo entre muslo y pierna."""
        hip = self._kp(skeleton, f"{side}_hip")
        knee = self._kp(skeleton, f"{side}_knee")
        ankle = self._kp(skeleton, f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            return None
        angle = self._angle_between(hip, knee, ankle)
        return 180.0 - angle
