# llm_rules/__init__.py
# Task C — Extracción de reglas ergonómicas desde PDF con Gemma/Ollama
# Depende de: schemas
#
# Implementar en este orden:
#   1. gpu_manager.py   → GPUManager (release_llm_vram con keep_alive=0)
#   2. pdf_extractor.py → PDFExtractor (PyMuPDF, chunking con overlap)
#   3. rule_cache.py    → RuleCache (SHA-256, guarda en .ergo_cache/)
#   4. rule_extractor.py→ RuleExtractor (orquestador, SIEMPRE libera VRAM en finally)
#
# CRÍTICO: RuleExtractor.extract() DEBE llamar GPUManager.release_llm_vram()
# en un bloque finally para liberar la GPU antes de que pose_pipeline la use.
#
# Ver CLAUDE.md para la implementación detallada.

from .rule_extractor import RuleExtractor
from .gpu_manager import GPUManager

__all__ = ["RuleExtractor", "GPUManager"]
