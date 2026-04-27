# advanced_pipeline/__init__.py
# Task F — Pipeline avanzado GPU (GVHMR + STRIDE + HumanMM)
# Depende de: schemas, pose_pipeline (interfaz pública)
# Implementar después de Task A (pose_pipeline completado).
#
# Drop-in replacement de PosePipeline cuando ProcessingMode.GPU_ENHANCED está activo.
# SIEMPRE retorna SkeletonSequence válido — fallback a PosePipeline base si los
# modelos avanzados no están instalados.
#
# Ver CLAUDE.md para la implementación detallada.

from .pipeline import AdvancedPosePipeline

__all__ = ["AdvancedPosePipeline"]
