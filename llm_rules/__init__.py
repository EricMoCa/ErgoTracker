"""
llm_rules — Extracción de reglas ergonómicas desde PDF con Gemma/Ollama.
Depende de: schemas
"""
from .gpu_manager import GPUManager
from .rule_extractor import RuleExtractor

__all__ = ["RuleExtractor", "GPUManager"]
