"""
ergo_engine/owas.py
Ovako Working Posture Analysis System (OWAS).
Referencia: Karhu et al. (1977). Journal of Occupational Health.
"""
from loguru import logger

from schemas import JointAngles, OWASCode, RiskLevel


# ---------------------------------------------------------------------------
# OWAS Action Category Table
# back (1-4) x arms (1-3) → action_level
# This is the simplified OWAS action table (load is factored separately)
# ---------------------------------------------------------------------------
# Standard OWAS action table: rows=back(1-4), cols=arms(1-3)
# Values 1-4 indicate action priority
ACTION_TABLE = [
    #arms:  1    2    3
    [       1,   1,   1],  # back=1
    [       1,   1,   1],  # back=2
    [       1,   1,   2],  # back=3
    [       2,   2,   2],  # back=4
]

# Full OWAS table with legs: back(1-4) x legs(1-7) x arms(1-3) → action
# Simplified version based on the original publication
FULL_ACTION_TABLE = {
    # (back, legs, arms) -> action_level
    (1, 1, 1): 1, (1, 1, 2): 1, (1, 1, 3): 1,
    (1, 2, 1): 1, (1, 2, 2): 1, (1, 2, 3): 1,
    (1, 3, 1): 1, (1, 3, 2): 1, (1, 3, 3): 1,
    (1, 4, 1): 2, (1, 4, 2): 2, (1, 4, 3): 2,
    (1, 5, 1): 2, (1, 5, 2): 2, (1, 5, 3): 2,
    (1, 6, 1): 1, (1, 6, 2): 1, (1, 6, 3): 1,
    (1, 7, 1): 1, (1, 7, 2): 1, (1, 7, 3): 1,

    (2, 1, 1): 2, (2, 1, 2): 2, (2, 1, 3): 3,
    (2, 2, 1): 2, (2, 2, 2): 2, (2, 2, 3): 3,
    (2, 3, 1): 2, (2, 3, 2): 2, (2, 3, 3): 3,
    (2, 4, 1): 3, (2, 4, 2): 3, (2, 4, 3): 4,
    (2, 5, 1): 3, (2, 5, 2): 3, (2, 5, 3): 4,
    (2, 6, 1): 2, (2, 6, 2): 2, (2, 6, 3): 3,
    (2, 7, 1): 2, (2, 7, 2): 2, (2, 7, 3): 3,

    (3, 1, 1): 2, (3, 1, 2): 2, (3, 1, 3): 3,
    (3, 2, 1): 2, (3, 2, 2): 2, (3, 2, 3): 3,
    (3, 3, 1): 2, (3, 3, 2): 3, (3, 3, 3): 3,
    (3, 4, 1): 3, (3, 4, 2): 3, (3, 4, 3): 4,
    (3, 5, 1): 3, (3, 5, 2): 3, (3, 5, 3): 4,
    (3, 6, 1): 2, (3, 6, 2): 3, (3, 6, 3): 3,
    (3, 7, 1): 2, (3, 7, 2): 2, (3, 7, 3): 3,

    (4, 1, 1): 2, (4, 1, 2): 3, (4, 1, 3): 4,
    (4, 2, 1): 3, (4, 2, 2): 3, (4, 2, 3): 4,
    (4, 3, 1): 3, (4, 3, 2): 3, (4, 3, 3): 4,
    (4, 4, 1): 4, (4, 4, 2): 4, (4, 4, 3): 4,
    (4, 5, 1): 4, (4, 5, 2): 4, (4, 5, 3): 4,
    (4, 6, 1): 3, (4, 6, 2): 3, (4, 6, 3): 4,
    (4, 7, 1): 3, (4, 7, 2): 3, (4, 7, 3): 4,
}


class OWASAnalyzer:
    """
    Ovako Working Posture Analysis System (OWAS).
    Referencia: Karhu et al. (1977). Journal of Occupational Health.
    """

    def analyze(self, angles: JointAngles, load_kg: float = 0) -> OWASCode:
        """
        Clasifica postura en 4 categorías + nivel de acción.
        load_kg: peso cargado (0=sin carga, <10=ligero, <20=moderado, >=20=pesado)
        """
        back_code = self._classify_back(
            trunk_flexion=angles.trunk_flexion or 0.0,
            trunk_rotation=angles.trunk_rotation or 0.0,
        )
        arms_code = self._classify_arms(
            shoulder_elev_l=angles.shoulder_elevation_left or 0.0,
            shoulder_elev_r=angles.shoulder_elevation_right or 0.0,
        )
        legs_code = self._classify_legs(
            knee_flexion_l=angles.knee_flexion_left or 0.0,
            knee_flexion_r=angles.knee_flexion_right or 0.0,
        )
        load_code = self._classify_load(load_kg)

        # Get action level from full table, fallback to simple ACTION_TABLE
        key = (back_code, legs_code, arms_code)
        action_level = FULL_ACTION_TABLE.get(key)
        if action_level is None:
            # Fallback to simple back x arms table
            b_idx = min(back_code - 1, 3)
            a_idx = min(arms_code - 1, 2)
            action_level = ACTION_TABLE[b_idx][a_idx]

        # Load modifier: heavy loads increase action level
        if load_code == 3 and action_level < 4:
            action_level = min(action_level + 1, 4)
        elif load_code == 2 and action_level < 3:
            action_level = min(action_level + 1, 4)

        risk_level = self._risk_level_from_action(action_level)

        return OWASCode(
            back_code=back_code,
            arms_code=arms_code,
            legs_code=legs_code,
            load_code=load_code,
            action_level=action_level,
            risk_level=risk_level,
        )

    def _classify_back(self, trunk_flexion: float, trunk_rotation: float) -> int:
        """
        1: erecto (0-20°)
        2: inclinado (>20°)
        3: rotado (>10° rotación, <20° flexión)
        4: inclinado+rotado (>20° flexión + >10° rotación)
        """
        is_inclined = trunk_flexion > 20.0
        is_rotated = trunk_rotation > 10.0

        if is_inclined and is_rotated:
            return 4
        elif is_rotated:
            return 3
        elif is_inclined:
            return 2
        else:
            return 1

    def _classify_arms(self, shoulder_elev_l: float, shoulder_elev_r: float) -> int:
        """
        1: ambos brazos por debajo del nivel del hombro (<90°)
        2: un brazo a nivel o por encima del hombro (>=90°)
        3: ambos brazos a nivel o por encima del hombro (>=90°)
        """
        left_above = shoulder_elev_l >= 90.0
        right_above = shoulder_elev_r >= 90.0

        if left_above and right_above:
            return 3
        elif left_above or right_above:
            return 2
        else:
            return 1

    def _classify_legs(self, knee_flexion_l: float, knee_flexion_r: float) -> int:
        """
        1: sentado
        2: de pie, ambas piernas extendidas
        3: de pie, peso en una pierna
        4: de pie, rodillas flexionadas (ambas)
        5: de pie, rodilla(s) muy flexionadas
        6: arrodillado
        7: caminando
        Simplificación basada en la flexión de rodilla disponible.
        """
        avg_knee = (knee_flexion_l + knee_flexion_r) / 2.0
        max_knee = max(knee_flexion_l, knee_flexion_r)

        if avg_knee > 90.0:
            return 6  # arrodillado
        elif max_knee > 60.0:
            return 5  # rodillas muy flexionadas
        elif avg_knee > 30.0:
            return 4  # rodillas flexionadas
        elif max_knee > 15.0:
            return 3  # peso en una pierna
        else:
            return 2  # de pie, piernas extendidas

    def _classify_load(self, load_kg: float) -> int:
        """
        1: <10 kg
        2: 10-20 kg
        3: >20 kg
        """
        if load_kg < 10.0:
            return 1
        elif load_kg <= 20.0:
            return 2
        else:
            return 3

    def _risk_level_from_action(self, action_level: int) -> RiskLevel:
        """
        1: NEGLIGIBLE
        2: LOW
        3: HIGH
        4: VERY_HIGH
        """
        mapping = {
            1: RiskLevel.NEGLIGIBLE,
            2: RiskLevel.LOW,
            3: RiskLevel.HIGH,
            4: RiskLevel.VERY_HIGH,
        }
        return mapping.get(action_level, RiskLevel.VERY_HIGH)
