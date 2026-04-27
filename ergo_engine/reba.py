"""
ergo_engine/reba.py
Rapid Entire Body Assessment (REBA).
Referencia: Hignett & McAtamney (2000). Applied Ergonomics, 31(2), 201-205.
"""
from loguru import logger

from schemas import JointAngles, REBAScore, RiskLevel


# ---------------------------------------------------------------------------
# REBA Table A: Neck (1-3) x Trunk (1-5) x Legs (1-4) → score
# ---------------------------------------------------------------------------
# Index: TABLE_A[neck_score-1][trunk_score-1][legs_score-1]
# Neck scores: 1, 2, 3
# Trunk scores: 1, 2, 3, 4, 5
# Legs scores: 1, 2, 3, 4
TABLE_A = [
    # neck=1
    [
        [1, 2, 3, 4],   # trunk=1, legs=1,2,3,4
        [2, 3, 4, 5],   # trunk=2
        [2, 4, 5, 6],   # trunk=3
        [3, 5, 6, 7],   # trunk=4
        [4, 6, 7, 8],   # trunk=5
    ],
    # neck=2
    [
        [1, 3, 4, 5],
        [2, 4, 5, 6],
        [3, 5, 6, 7],
        [4, 6, 7, 8],
        [5, 7, 8, 9],
    ],
    # neck=3
    [
        [3, 4, 5, 6],
        [3, 5, 6, 7],
        [4, 6, 7, 8],
        [5, 7, 8, 9],
        [6, 8, 9, 9],
    ],
]

# ---------------------------------------------------------------------------
# REBA Table B: Upper Arm (1-6) x Lower Arm (1-2) x Wrist (1-3) → score
# ---------------------------------------------------------------------------
# Index: TABLE_B[upper_arm-1][lower_arm-1][wrist-1]
TABLE_B = [
    # upper_arm=1
    [
        [1, 2, 2],   # lower_arm=1, wrist=1,2,3
        [1, 2, 3],   # lower_arm=2
    ],
    # upper_arm=2
    [
        [1, 2, 3],
        [2, 3, 4],
    ],
    # upper_arm=3
    [
        [3, 4, 5],
        [4, 5, 5],
    ],
    # upper_arm=4
    [
        [4, 5, 5],
        [5, 6, 7],
    ],
    # upper_arm=5
    [
        [6, 7, 8],
        [7, 8, 8],
    ],
    # upper_arm=6
    [
        [7, 8, 8],
        [8, 9, 9],
    ],
]

# ---------------------------------------------------------------------------
# REBA Table C: Group A score (1-12) x Group B score (1-12) → total
# ---------------------------------------------------------------------------
# Index: TABLE_C[score_a-1][score_b-1]
TABLE_C = [
    #  B: 1   2   3   4   5   6   7   8   9  10  11  12
    [   1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],  # A=1
    [   1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],  # A=2
    [   2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],  # A=3
    [   3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],  # A=4
    [   4,  4,  4,  5,  6,  7,  8,  8,  9,  9, 10, 10],  # A=5
    [   6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 11, 11],  # A=6
    [   7,  7,  7,  8,  9,  9,  9, 10, 11, 11, 11, 12],  # A=7
    [   8,  8,  8,  9, 10, 10, 10, 10, 11, 11, 12, 12],  # A=8
    [   9,  9,  9, 10, 10, 10, 11, 11, 12, 12, 12, 12],  # A=9
    [  10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],  # A=10
    [  11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],  # A=11
    [  12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],  # A=12
]


class REBAAnalyzer:
    """
    Rapid Entire Body Assessment (REBA).
    Referencia: Hignett & McAtamney (2000). Applied Ergonomics, 31(2), 201-205.
    """

    def analyze(self, angles: JointAngles) -> REBAScore:
        """
        Calcula la puntuación REBA completa.
        Retorna REBAScore con todos los sub-scores y nivel de riesgo.
        """
        # --- Group A ---
        neck_score = self._score_neck(
            neck_flexion=angles.neck_flexion or 0.0,
            lateral=angles.neck_lateral_bending or 0.0,
        )
        trunk_score = self._score_trunk(
            trunk_flexion=angles.trunk_flexion or 0.0,
            lateral=angles.trunk_lateral_bending or 0.0,
        )
        legs_score = self._score_legs(
            knee_flexion=max(
                angles.knee_flexion_left or 0.0,
                angles.knee_flexion_right or 0.0,
            )
        )

        # Clamp indices
        n_idx = min(neck_score - 1, 2)  # max neck index = 2 (score 1-3)
        t_idx = min(trunk_score - 1, 4)  # max trunk index = 4 (score 1-5)
        l_idx = min(legs_score - 1, 3)  # max legs index = 3 (score 1-4)

        n_idx = max(n_idx, 0)
        t_idx = max(t_idx, 0)
        l_idx = max(l_idx, 0)

        group_a = TABLE_A[n_idx][t_idx][l_idx]

        # --- Group B ---
        upper_arm_score = self._score_upper_arm(
            shoulder_elevation=max(
                angles.shoulder_elevation_left or 0.0,
                angles.shoulder_elevation_right or 0.0,
            )
        )
        lower_arm_score = self._score_lower_arm(
            elbow_flexion=max(
                angles.elbow_flexion_left or 80.0,
                angles.elbow_flexion_right or 80.0,
            )
        )
        wrist_score = self._score_wrist(
            wrist_flexion=max(
                angles.wrist_flexion_left or 0.0,
                angles.wrist_flexion_right or 0.0,
            ),
            wrist_deviation=max(
                angles.wrist_deviation_left or 0.0,
                angles.wrist_deviation_right or 0.0,
            ),
        )

        ua_idx = min(upper_arm_score - 1, 5)
        la_idx = min(lower_arm_score - 1, 1)
        w_idx = min(wrist_score - 1, 2)

        ua_idx = max(ua_idx, 0)
        la_idx = max(la_idx, 0)
        w_idx = max(w_idx, 0)

        group_b = TABLE_B[ua_idx][la_idx][w_idx]

        # --- Table C ---
        a_idx = min(group_a - 1, 11)
        b_idx = min(group_b - 1, 11)
        a_idx = max(a_idx, 0)
        b_idx = max(b_idx, 0)

        total = TABLE_C[a_idx][b_idx]
        total = max(1, min(15, total))

        risk_level = self._risk_level_from_score(total)

        return REBAScore(
            total=total,
            risk_level=risk_level,
            neck_score=neck_score,
            trunk_score=trunk_score,
            legs_score=legs_score,
            upper_arm_score=upper_arm_score,
            lower_arm_score=lower_arm_score,
            wrist_score=wrist_score,
            group_a=group_a,
            group_b=group_b,
            activity_score=0,
        )

    def _score_neck(self, neck_flexion: float, lateral: float = 0.0) -> int:
        """
        1: 0-20° flexión
        2: >20° flexión o extensión
        +1: si hay torsión o inclinación lateral
        """
        if 0.0 <= neck_flexion <= 20.0:
            score = 1
        else:
            score = 2
        if lateral > 10.0:
            score += 1
        return score

    def _score_trunk(self, trunk_flexion: float, lateral: float = 0.0) -> int:
        """
        1: 0° (erecto)
        2: 0-20° flexión o 0-20° extensión
        3: 20-60° flexión o >20° extensión
        4: >60° flexión
        +1: si hay torsión o inclinación lateral
        """
        if trunk_flexion <= 5.0:
            score = 1
        elif trunk_flexion <= 20.0:
            score = 2
        elif trunk_flexion <= 60.0:
            score = 3
        else:
            score = 4
        if lateral > 10.0:
            score += 1
        return score

    def _score_legs(self, knee_flexion: float) -> int:
        """
        1: soporte bilateral, caminando o sentado
        2: soporte unilateral o postura inestable
        +1: 30-60° flexión de rodilla
        +2: >60° flexión de rodilla
        """
        score = 1  # assume bilateral support by default
        if 30.0 <= knee_flexion <= 60.0:
            score += 1
        elif knee_flexion > 60.0:
            score += 2
        return score

    def _score_upper_arm(self, shoulder_elevation: float) -> int:
        """
        1: 20° extensión a 20° flexión
        2: >20° extensión o 20-45° flexión
        3: 45-90° flexión
        4: >90° flexión
        +1: si hombro elevado o brazo abducido
        -1: si hay apoyo del brazo
        """
        if shoulder_elevation <= 20.0:
            score = 1
        elif shoulder_elevation <= 45.0:
            score = 2
        elif shoulder_elevation <= 90.0:
            score = 3
        else:
            score = 4
        return score

    def _score_lower_arm(self, elbow_flexion: float) -> int:
        """
        1: 60-100° flexión
        2: <60° o >100° flexión
        """
        if 60.0 <= elbow_flexion <= 100.0:
            return 1
        return 2

    def _score_wrist(self, wrist_flexion: float, wrist_deviation: float = 0.0) -> int:
        """
        1: 0-15° flexión/extensión
        2: >15° flexión/extensión
        +1: si hay desviación o torsión
        """
        if wrist_flexion <= 15.0:
            score = 1
        else:
            score = 2
        if wrist_deviation > 10.0:
            score += 1
        return score

    def _risk_level_from_score(self, score: int) -> RiskLevel:
        """
        1:       NEGLIGIBLE
        2-3:     LOW
        4-7:     MEDIUM
        8-10:    HIGH
        11-15:   VERY_HIGH
        """
        if score == 1:
            return RiskLevel.NEGLIGIBLE
        elif score <= 3:
            return RiskLevel.LOW
        elif score <= 7:
            return RiskLevel.MEDIUM
        elif score <= 10:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
