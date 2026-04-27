"""
ergo_engine/rula.py
Rapid Upper Limb Assessment (RULA).
Referencia: McAtamney & Corlett (1993). Applied Ergonomics, 24(2), 91-99.
"""
from loguru import logger

from schemas import JointAngles, RULAScore, RiskLevel


# ---------------------------------------------------------------------------
# RULA Table A: Upper Arm (1-4) x Lower Arm (1-2) x Wrist (1-4) → score
# Wrist twist adds 1 (handled separately)
# ---------------------------------------------------------------------------
# Index: TABLE_A[upper_arm-1][lower_arm-1][wrist-1]
TABLE_A = {
    # upper_arm=1
    (1, 1, 1): 1, (1, 1, 2): 2, (1, 1, 3): 2, (1, 1, 4): 2,
    (1, 2, 1): 2, (1, 2, 2): 2, (1, 2, 3): 3, (1, 2, 4): 3,
    # upper_arm=2
    (2, 1, 1): 2, (2, 1, 2): 2, (2, 1, 3): 3, (2, 1, 4): 3,
    (2, 2, 1): 2, (2, 2, 2): 3, (2, 2, 3): 3, (2, 2, 4): 3,
    # upper_arm=3
    (3, 1, 1): 2, (3, 1, 2): 3, (3, 1, 3): 3, (3, 1, 4): 3,
    (3, 2, 1): 2, (3, 2, 2): 3, (3, 2, 3): 4, (3, 2, 4): 4,
    # upper_arm=4
    (4, 1, 1): 3, (4, 1, 2): 3, (4, 1, 3): 4, (4, 1, 4): 4,
    (4, 2, 1): 3, (4, 2, 2): 4, (4, 2, 3): 4, (4, 2, 4): 5,
}

# ---------------------------------------------------------------------------
# RULA Table B: Neck (1-4) x Trunk (1-5) x Legs (1-2) → score
# ---------------------------------------------------------------------------
TABLE_B = {
    # neck=1
    (1, 1, 1): 1, (1, 1, 2): 3,
    (1, 2, 1): 2, (1, 2, 2): 3,
    (1, 3, 1): 3, (1, 3, 2): 4,
    (1, 4, 1): 5, (1, 4, 2): 5,
    (1, 5, 1): 6, (1, 5, 2): 6,
    # neck=2
    (2, 1, 1): 2, (2, 1, 2): 3,
    (2, 2, 1): 2, (2, 2, 2): 3,
    (2, 3, 1): 4, (2, 3, 2): 5,
    (2, 4, 1): 5, (2, 4, 2): 5,
    (2, 5, 1): 6, (2, 5, 2): 7,
    # neck=3
    (3, 1, 1): 3, (3, 1, 2): 3,
    (3, 2, 1): 3, (3, 2, 2): 4,
    (3, 3, 1): 4, (3, 3, 2): 5,
    (3, 4, 1): 5, (3, 4, 2): 6,
    (3, 5, 1): 6, (3, 5, 2): 7,
    # neck=4
    (4, 1, 1): 4, (4, 1, 2): 4,
    (4, 2, 1): 4, (4, 2, 2): 4,
    (4, 3, 1): 4, (4, 3, 2): 5,
    (4, 4, 1): 6, (4, 4, 2): 6,
    (4, 5, 1): 7, (4, 5, 2): 7,
}

# ---------------------------------------------------------------------------
# RULA Table C: Score A (1-8) x Score B (1-7) → final score
# ---------------------------------------------------------------------------
TABLE_C = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 5,
    (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 4, (2, 6): 5, (2, 7): 5,
    (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 4, (3, 6): 5, (3, 7): 6,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 6,
    (5, 1): 4, (5, 2): 4, (5, 3): 4, (5, 4): 5, (5, 5): 6, (5, 6): 7, (5, 7): 7,
    (6, 1): 4, (6, 2): 4, (6, 3): 5, (6, 4): 6, (6, 5): 6, (6, 6): 7, (6, 7): 7,
    (7, 1): 5, (7, 2): 5, (7, 3): 6, (7, 4): 6, (7, 5): 7, (7, 6): 7, (7, 7): 7,
    (8, 1): 5, (8, 2): 5, (8, 3): 6, (8, 4): 7, (8, 5): 7, (8, 6): 7, (8, 7): 7,
}


class RULAAnalyzer:
    """
    Rapid Upper Limb Assessment (RULA).
    Referencia: McAtamney & Corlett (1993). Applied Ergonomics, 24(2), 91-99.
    """

    def analyze(self, angles: JointAngles) -> RULAScore:
        """Calcula la puntuación RULA completa."""
        # --- Group A: upper limb ---
        upper_arm_score = self._score_upper_arm_rula(
            elevation=max(
                angles.shoulder_elevation_left or 0.0,
                angles.shoulder_elevation_right or 0.0,
            )
        )
        lower_arm_score = self._score_lower_arm_rula(
            flexion=max(
                angles.elbow_flexion_left or 80.0,
                angles.elbow_flexion_right or 80.0,
            )
        )
        wrist_score = self._score_wrist_rula(
            flexion=max(
                angles.wrist_flexion_left or 0.0,
                angles.wrist_flexion_right or 0.0,
            )
        )
        wrist_twist_score = 1  # neutral by default

        # Look up Table A
        ua = min(max(upper_arm_score, 1), 4)
        la = min(max(lower_arm_score, 1), 2)
        w = min(max(wrist_score, 1), 4)
        group_a = TABLE_A.get((ua, la, w), TABLE_A.get((min(ua, 4), min(la, 2), min(w, 4)), 4))
        group_a = group_a + wrist_twist_score - 1  # wrist twist: 1=normal, 2=twisted

        # --- Group B: neck/trunk/legs ---
        neck_score = self._score_neck_rula(
            flexion=angles.neck_flexion or 0.0
        )
        trunk_score = self._score_trunk_rula(
            flexion=angles.trunk_flexion or 0.0
        )
        legs_score = 1  # bilateral support by default

        # Look up Table B
        n = min(max(neck_score, 1), 4)
        t = min(max(trunk_score, 1), 5)
        le = min(max(legs_score, 1), 2)
        group_b = TABLE_B.get((n, t, le), 4)

        # Muscle use and force/load adjustments (default 0)
        muscle_use_score = 0
        force_load_score = 0

        score_a = min(max(group_a + muscle_use_score + force_load_score, 1), 8)
        score_b = min(max(group_b + muscle_use_score + force_load_score, 1), 7)

        # Look up Table C
        total = TABLE_C.get((score_a, score_b), 7)
        total = max(1, min(7, total))

        risk_level = self._risk_level_from_score(total)

        return RULAScore(
            total=total,
            risk_level=risk_level,
            upper_arm_score=upper_arm_score,
            lower_arm_score=lower_arm_score,
            wrist_score=wrist_score,
            wrist_twist_score=wrist_twist_score,
            neck_score=neck_score,
            trunk_score=trunk_score,
            legs_score=legs_score,
            group_a=group_a,
            group_b=group_b,
            muscle_use_score=muscle_use_score,
            force_load_score=force_load_score,
        )

    def _score_upper_arm_rula(self, elevation: float) -> int:
        """
        1: 20° extensión a 20° flexión
        2: >20° extensión o 20-45° flexión
        3: 45-90° flexión
        4: >90° flexión
        """
        if elevation <= 20.0:
            return 1
        elif elevation <= 45.0:
            return 2
        elif elevation <= 90.0:
            return 3
        else:
            return 4

    def _score_lower_arm_rula(self, flexion: float) -> int:
        """
        1: 60-100° flexión
        2: <60° o >100° flexión
        """
        if 60.0 <= flexion <= 100.0:
            return 1
        return 2

    def _score_wrist_rula(self, flexion: float) -> int:
        """
        1: posición neutra (0-15°)
        2: 15-30° flexión/extensión
        3: >30° flexión/extensión
        4: desviación marcada (handled via wrist_twist)
        """
        if flexion <= 15.0:
            return 1
        elif flexion <= 30.0:
            return 2
        else:
            return 3

    def _score_neck_rula(self, flexion: float) -> int:
        """
        1: 0-10° flexión
        2: 10-20° flexión
        3: >20° flexión
        4: extensión
        """
        if flexion < 0.0:  # extension
            return 4
        elif flexion <= 10.0:
            return 1
        elif flexion <= 20.0:
            return 2
        else:
            return 3

    def _score_trunk_rula(self, flexion: float) -> int:
        """
        1: erecto / sentado apoyado
        2: 0-20° flexión
        3: 20-60° flexión
        4: >60° flexión
        5: extensión marcada (no en tabla original, se usa 4)
        """
        if flexion <= 5.0:
            return 1
        elif flexion <= 20.0:
            return 2
        elif flexion <= 60.0:
            return 3
        else:
            return 4

    def _risk_level_from_score(self, score: int) -> RiskLevel:
        """
        1-2: NEGLIGIBLE
        3-4: LOW
        5-6: MEDIUM
        7:   VERY_HIGH
        """
        if score <= 2:
            return RiskLevel.NEGLIGIBLE
        elif score <= 4:
            return RiskLevel.LOW
        elif score <= 6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.VERY_HIGH
