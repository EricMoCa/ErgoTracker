from typing import Optional

from loguru import logger

from schemas import (
    SkeletonSequence, JointAngles, FrameErgonomicScore, RiskLevel,
)
from .angles import JointAngleCalculator
from .reba import REBAAnalyzer
from .rula import RULAAnalyzer
from .owas import OWASAnalyzer
from .llm_rule_analyzer import LLMRuleAnalyzer


class ErgoEngine:
    """
    Motor de análisis ergonómico. Orquesta todos los analizadores.
    Entrada: SkeletonSequence
    Salida: list[FrameErgonomicScore]
    """

    def __init__(
        self,
        methods: list[str] = ["REBA"],
        rules_json_path: Optional[str] = None,
        load_kg: float = 0.0,
    ) -> None:
        self.methods = methods
        self.load_kg = load_kg
        self.angle_calculator = JointAngleCalculator()

        self.reba: Optional[REBAAnalyzer] = REBAAnalyzer() if "REBA" in methods else None
        self.rula: Optional[RULAAnalyzer] = RULAAnalyzer() if "RULA" in methods else None
        self.owas: Optional[OWASAnalyzer] = OWASAnalyzer() if "OWAS" in methods else None
        self.llm_analyzer: Optional[LLMRuleAnalyzer] = None

        if "LLM" in methods and rules_json_path:
            self.llm_analyzer = LLMRuleAnalyzer()
            self.llm_analyzer.load_rules(rules_json_path)

    def analyze(self, skeleton_sequence: SkeletonSequence) -> list[FrameErgonomicScore]:
        """Analiza cada frame del skeleton_sequence y retorna scores."""
        results: list[FrameErgonomicScore] = []

        for skeleton in skeleton_sequence.frames:
            angles = self.angle_calculator.calculate(skeleton)
            frame_score = FrameErgonomicScore(
                frame_idx=skeleton.frame_idx,
                timestamp_s=skeleton.timestamp_s,
            )

            if self.reba:
                try:
                    frame_score.reba = self.reba.analyze(angles)
                except Exception as e:
                    logger.warning(f"REBA falló en frame {skeleton.frame_idx}: {e}")

            if self.rula:
                try:
                    frame_score.rula = self.rula.analyze(angles)
                except Exception as e:
                    logger.warning(f"RULA falló en frame {skeleton.frame_idx}: {e}")

            if self.owas:
                try:
                    frame_score.owas = self.owas.analyze(angles, load_kg=self.load_kg)
                except Exception as e:
                    logger.warning(f"OWAS falló en frame {skeleton.frame_idx}: {e}")

            if self.llm_analyzer:
                try:
                    frame_score.rule_violations = self.llm_analyzer.evaluate(angles)
                except Exception as e:
                    logger.warning(f"LLM rules falló en frame {skeleton.frame_idx}: {e}")

            frame_score.overall_risk = self._compute_overall_risk(frame_score)
            results.append(frame_score)

        return results

    def _compute_overall_risk(self, frame_score: FrameErgonomicScore) -> RiskLevel:
        """El riesgo global es el máximo entre todos los métodos del frame."""
        risk_order = [
            RiskLevel.NEGLIGIBLE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.VERY_HIGH,
        ]

        candidates: list[RiskLevel] = [RiskLevel.NEGLIGIBLE]

        if frame_score.reba:
            candidates.append(frame_score.reba.risk_level)
        if frame_score.rula:
            candidates.append(frame_score.rula.risk_level)
        if frame_score.owas:
            candidates.append(frame_score.owas.risk_level)
        for v in frame_score.rule_violations:
            candidates.append(v.rule.risk_level)

        return max(candidates, key=lambda r: risk_order.index(r))
