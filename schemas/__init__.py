# schemas/__init__.py
# Task 0 — FOUNDATION: contratos de datos compartidos
# Todos los módulos del proyecto importan ÚNICAMENTE de schemas.
#
# Implementar en este orden:
#   1. video.py      → VideoInput, ProcessingMode
#   2. skeleton.py   → Keypoint3D, Skeleton3D, SkeletonSequence, KEYPOINT_NAMES
#   3. angles.py     → JointAngles
#   4. ergonomic.py  → RiskLevel, ErgonomicRule, RuleViolation, REBAScore,
#                      RULAScore, OWASCode, FrameErgonomicScore,
#                      ReportSummary, AnalysisReport
#
# Ver CLAUDE.md para los modelos Pydantic completos.

from .video import VideoInput, ProcessingMode
from .skeleton import Keypoint3D, Skeleton3D, SkeletonSequence, KEYPOINT_NAMES
from .angles import JointAngles
from .ergonomic import (
    RiskLevel,
    ErgonomicRule,
    RuleViolation,
    REBAScore,
    RULAScore,
    OWASCode,
    FrameErgonomicScore,
    ReportSummary,
    AnalysisReport,
)

__all__ = [
    "VideoInput",
    "ProcessingMode",
    "Keypoint3D",
    "Skeleton3D",
    "SkeletonSequence",
    "KEYPOINT_NAMES",
    "JointAngles",
    "RiskLevel",
    "ErgonomicRule",
    "RuleViolation",
    "REBAScore",
    "RULAScore",
    "OWASCode",
    "FrameErgonomicScore",
    "ReportSummary",
    "AnalysisReport",
]
