# ergo_engine/__init__.py
# Task B — Motor de análisis ergonómico
# Depende de: schemas
#
# Implementar en este orden:
#   1. angles.py            → JointAngleCalculator
#   2. reba.py              → REBAAnalyzer
#   3. rula.py              → RULAAnalyzer
#   4. owas.py              → OWASAnalyzer
#   5. llm_rule_analyzer.py → LLMRuleAnalyzer
#   6. engine.py            → ErgoEngine (orquestador)
#
# Ver CLAUDE.md para la implementación detallada.

from .engine import ErgoEngine
from .angles import JointAngleCalculator

__all__ = ["ErgoEngine", "JointAngleCalculator"]
