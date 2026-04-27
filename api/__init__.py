# api/__init__.py
# Task D — Backend FastAPI (orquestador)
# Depende de: schemas, pose_pipeline, ergo_engine, llm_rules, reports
# IMPLEMENTAR ÚLTIMO — requiere que Tasks A, B, C y E estén completos.
#
# Ver CLAUDE.md para la implementación detallada.

from .main import app

__all__ = ["app"]
