import json
import operator
from pathlib import Path
from typing import Optional

from loguru import logger

from schemas import JointAngles, ErgonomicRule, RuleViolation


class LLMRuleAnalyzer:
    """
    Evalúa reglas ergonómicas definidas en ErgonomicRule contra los ángulos medidos.
    Las reglas son generadas por llm_rules/ y cargadas desde un JSON.
    """

    OPS = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
    }

    def __init__(self) -> None:
        self._rules: list[ErgonomicRule] = []

    def load_rules(self, rules_json_path: str) -> None:
        """Carga reglas desde archivo JSON generado por llm_rules/."""
        path = Path(rules_json_path)
        if not path.exists():
            logger.warning(f"Archivo de reglas no encontrado: {rules_json_path}")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "rules" in data:
            rules_data = data["rules"]
        elif isinstance(data, list):
            rules_data = data
        else:
            logger.error(f"Formato de reglas desconocido en {rules_json_path}")
            return
        self._rules = [ErgonomicRule(**r) for r in rules_data]
        logger.info(f"Cargadas {len(self._rules)} reglas desde {rules_json_path}")

    def evaluate(self, angles: JointAngles) -> list[RuleViolation]:
        """
        Evalúa todas las reglas cargadas contra los ángulos del frame.
        """
        violations: list[RuleViolation] = []
        for rule in self._rules:
            angle_value: Optional[float] = getattr(angles, rule.joint, None)
            if angle_value is None:
                continue
            op_str, threshold = self._parse_condition(rule.condition)
            op_fn = self.OPS.get(op_str)
            if op_fn is None:
                logger.warning(f"Operador desconocido '{op_str}' en regla {rule.id}")
                continue
            if op_fn(angle_value, threshold):
                violations.append(
                    RuleViolation(
                        rule=rule,
                        measured_angle=angle_value,
                        threshold=threshold,
                    )
                )
        return violations

    def _parse_condition(self, condition: str) -> tuple[str, float]:
        """
        Parsea "angle > 60" → (">", 60.0)
        La variable siempre es "angle".
        """
        condition = condition.strip()
        for op in (">=", "<=", "==", ">", "<"):
            if op in condition:
                parts = condition.split(op)
                threshold_str = parts[1].strip().replace("angle", "").strip()
                return op, float(threshold_str)
        raise ValueError(f"No se pudo parsear condición: {condition}")

    @property
    def rules(self) -> list[ErgonomicRule]:
        return list(self._rules)
