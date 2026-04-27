# reports/__init__.py
# Task E — Generador de reportes PDF
# Depende de: schemas
#
# Implementar en este orden:
#   1. charts.py                       → ChartGenerator (matplotlib Agg, retorna base64 PNG)
#   2. templates/report_template.html  → Plantilla Jinja2 con estilos inline
#   3. generator.py                    → ReportGenerator (AnalysisReport → HTML → WeasyPrint → PDF)
#
# Ver CLAUDE.md para la implementación detallada y nota sobre WeasyPrint en Windows.

from .generator import ReportGenerator

__all__ = ["ReportGenerator"]
