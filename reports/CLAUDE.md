# Task E — reports/

## Objetivo
Implementar el generador de reportes PDF a partir de un `AnalysisReport`. El reporte debe ser profesional, visualmente claro y exportable. Completamente independiente del pipeline de visión y del LLM.

## Dependencia previa
**Requiere que `schemas/` esté completo.** Importa ÚNICAMENTE de `schemas`. NO depende de ningún otro módulo del proyecto. Puede desarrollarse en paralelo a Tasks A, B y C.

## Archivos a crear

```
reports/
├── __init__.py               # Expone: ReportGenerator
├── generator.py              # Orquestador principal
├── charts.py                 # Generación de gráficas (matplotlib)
├── templates/
│   └── report_template.html  # Plantilla Jinja2 para el reporte HTML→PDF
├── static/
│   └── report.css            # Estilos del reporte (inlineados en HTML)
├── conftest.py               # Fixtures con AnalysisReport sintético
└── tests/
    ├── test_generator.py
    └── test_charts.py
```

## Implementación por archivo

### charts.py
```python
import matplotlib
matplotlib.use("Agg")  # Sin display (headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import base64
from schemas import AnalysisReport, RiskLevel, FrameErgonomicScore

# Colores por nivel de riesgo
RISK_COLORS = {
    RiskLevel.NEGLIGIBLE: "#22C55E",   # verde
    RiskLevel.LOW:        "#86EFAC",   # verde claro
    RiskLevel.MEDIUM:     "#FCD34D",   # amarillo
    RiskLevel.HIGH:       "#F97316",   # naranja
    RiskLevel.VERY_HIGH:  "#EF4444",   # rojo
}

class ChartGenerator:
    """Genera gráficas para el reporte. Retorna imágenes como base64 PNG."""

    def risk_timeline(self, frame_scores: list[FrameErgonomicScore], fps: float = 25.0) -> str:
        """
        Gráfica de línea: eje X = tiempo (segundos), eje Y = score REBA/RULA.
        Colorear fondo por nivel de riesgo.
        Retorna imagen PNG en base64 para embeber en HTML.
        """

    def risk_distribution_pie(self, frame_scores: list[FrameErgonomicScore]) -> str:
        """
        Gráfica de torta: distribución de frames por nivel de riesgo.
        Retorna imagen PNG en base64.
        """

    def joint_angles_timeline(
        self,
        frame_scores: list[FrameErgonomicScore],
        joint_name: str = "trunk_flexion"
    ) -> str:
        """
        Gráfica de ángulo articular específico a lo largo del tiempo.
        Línea horizontal de umbral de riesgo si aplica.
        """

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convierte una figura matplotlib a string base64 PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()
```

### templates/report_template.html
Plantilla Jinja2 completa. Debe incluir:

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <style>
        /* Estilos inline — WeasyPrint no carga CSS externo por defecto */
        body { font-family: Arial, sans-serif; font-size: 11pt; color: #1f2937; }
        .cover { page-break-after: always; text-align: center; padding: 80px 40px; }
        .section-title { color: #1F4E79; border-bottom: 2px solid #2E75B6; padding-bottom: 4px; }
        .risk-badge { padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 10pt; }
        .risk-NEGLIGIBLE { background: #D1FAE5; color: #065F46; }
        .risk-LOW        { background: #D1FAE5; color: #065F46; }
        .risk-MEDIUM     { background: #FEF3C7; color: #92400E; }
        .risk-HIGH       { background: #FED7AA; color: #9A3412; }
        .risk-VERY_HIGH  { background: #FEE2E2; color: #991B1B; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; }
        th { background: #1F4E79; color: white; padding: 8px; font-size: 10pt; }
        td { padding: 6px 8px; border-bottom: 1px solid #E5E7EB; font-size: 10pt; }
        tr:nth-child(even) { background: #F0F7FF; }
        .chart-img { width: 100%; max-width: 600px; margin: 12px auto; display: block; }
        @page { margin: 2cm; @bottom-right { content: "Página " counter(page) " de " counter(pages); } }
    </style>
</head>
<body>

<!-- PORTADA -->
<div class="cover">
    <h1 style="color:#1F4E79; font-size:28pt;">Reporte de Análisis Ergonómico</h1>
    <h2 style="color:#2E75B6;">{{ report.video_path | basename }}</h2>
    <p>Generado: {{ report.created_at.strftime('%d/%m/%Y %H:%M') }}</p>
    <p>Persona analizada: {{ report.person_height_cm }} cm</p>
    <div class="risk-badge risk-{{ report.summary.max_risk_level }}">
        Riesgo máximo: {{ report.summary.max_risk_level }}
    </div>
</div>

<!-- RESUMEN EJECUTIVO -->
<h2 class="section-title">Resumen Ejecutivo</h2>
<table>
    <tr><th>Parámetro</th><th>Valor</th></tr>
    <tr><td>Duración del video</td><td>{{ "%.1f"|format(report.duration_s) }} segundos</td></tr>
    <tr><td>Frames analizados</td><td>{{ report.analyzed_frames }} / {{ report.total_frames }}</td></tr>
    <tr><td>Métodos aplicados</td><td>{{ report.methods_used | join(', ') }}</td></tr>
    {% if report.summary.max_reba_score %}
    <tr><td>Score REBA máximo</td><td>{{ report.summary.max_reba_score }} / 15</td></tr>
    {% endif %}
    <tr><td>Frames con riesgo ALTO o MUY ALTO</td><td>{{ "%.1f"|format(report.summary.pct_frames_high_risk * 100) }}%</td></tr>
</table>

<!-- GRÁFICAS -->
<h2 class="section-title">Evolución Temporal del Riesgo</h2>
<img class="chart-img" src="data:image/png;base64,{{ charts.risk_timeline }}"/>

<h2 class="section-title">Distribución de Riesgo</h2>
<img class="chart-img" src="data:image/png;base64,{{ charts.risk_pie }}"/>

<!-- VIOLACIONES DE REGLAS -->
{% if violations %}
<h2 class="section-title">Violaciones de Reglas Detectadas</h2>
<table>
    <tr><th>Regla</th><th>Articulación</th><th>Riesgo</th><th>Acción</th></tr>
    {% for v in violations %}
    <tr>
        <td>{{ v.rule.description }}</td>
        <td>{{ v.rule.joint }}</td>
        <td><span class="risk-badge risk-{{ v.rule.risk_level }}">{{ v.rule.risk_level }}</span></td>
        <td>{{ v.rule.action }}</td>
    </tr>
    {% endfor %}
</table>
{% endif %}

<!-- RECOMENDACIONES -->
<h2 class="section-title">Recomendaciones</h2>
{% for rec in report.summary.recommendations %}
<p>• {{ rec }}</p>
{% endfor %}

</body>
</html>
```

### generator.py
```python
from pathlib import Path
from datetime import datetime
import jinja2
import weasyprint
from schemas import AnalysisReport, RuleViolation, RiskLevel, FrameErgonomicScore
from .charts import ChartGenerator
from loguru import logger

class ReportGenerator:
    """
    Genera reportes PDF profesionales desde un AnalysisReport.
    Pipeline: AnalysisReport → Jinja2 HTML → WeasyPrint → PDF
    """

    def __init__(self):
        self.charts = ChartGenerator()
        template_dir = Path(__file__).parent / "templates"
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        # Filtro Jinja2 personalizado para basename de path
        self.env.filters["basename"] = lambda p: Path(p).name

    def generate(self, report: AnalysisReport, output_path: str) -> str:
        """
        Genera el reporte PDF completo.
        Args:
            report: AnalysisReport con todos los datos del análisis
            output_path: ruta de destino del PDF (ej: "reporte_2026.pdf")
        Returns:
            path absoluto del PDF generado
        """
        logger.info(f"Generando reporte PDF para {report.video_path}...")

        # 1. Generar gráficas
        charts_data = {
            "risk_timeline": self.charts.risk_timeline(report.frame_scores),
            "risk_pie": self.charts.risk_distribution_pie(report.frame_scores),
        }

        # 2. Recopilar violaciones únicas más frecuentes
        violations = self._get_top_violations(report.frame_scores, top_n=10)

        # 3. Renderizar HTML con Jinja2
        template = self.env.get_template("report_template.html")
        html_content = template.render(
            report=report,
            charts=charts_data,
            violations=violations,
        )

        # 4. Convertir HTML → PDF con WeasyPrint
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weasyprint.HTML(string=html_content).write_pdf(str(output_path))

        logger.success(f"Reporte generado: {output_path} ({output_path.stat().st_size // 1024} KB)")
        return str(output_path)

    def _get_top_violations(
        self,
        frame_scores: list[FrameErgonomicScore],
        top_n: int = 10
    ) -> list[RuleViolation]:
        """
        Extrae las N violaciones más frecuentes de toda la secuencia.
        Agrupa por rule.id y retorna una por regla (la de mayor ángulo medido).
        """
        from collections import defaultdict
        violations_by_rule: dict[str, list[RuleViolation]] = defaultdict(list)
        for frame in frame_scores:
            for v in frame.rule_violations:
                violations_by_rule[v.rule.id].append(v)

        # Ordenar por frecuencia y retornar la peor instancia de cada regla
        sorted_rules = sorted(
            violations_by_rule.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        return [
            max(violations, key=lambda v: v.measured_angle)
            for _, violations in sorted_rules[:top_n]
        ]
```

## Interfaz Pública (`__init__.py`)
```python
from .generator import ReportGenerator

__all__ = ["ReportGenerator"]
```

## Tests

### conftest.py
```python
import pytest
from datetime import datetime
from schemas import (
    AnalysisReport, FrameErgonomicScore, REBAScore, RiskLevel,
    ReportSummary, ErgonomicRule, RuleViolation
)

@pytest.fixture
def sample_report():
    """AnalysisReport sintético con 10 frames de datos."""
    frames = []
    for i in range(10):
        risk = RiskLevel.HIGH if i > 6 else RiskLevel.MEDIUM
        frames.append(FrameErgonomicScore(
            frame_idx=i,
            timestamp_s=float(i),
            reba=REBAScore(
                total=8 if i > 6 else 5,
                risk_level=risk,
                neck_score=2, trunk_score=3, legs_score=2,
                upper_arm_score=2, lower_arm_score=1, wrist_score=2,
                group_a=4, group_b=3
            ),
            overall_risk=risk
        ))
    return AnalysisReport(
        id="test-001",
        created_at=datetime.now(),
        video_path="/videos/operario_tarea_A.mp4",
        duration_s=10.0,
        total_frames=300,
        analyzed_frames=10,
        person_height_cm=170.0,
        methods_used=["REBA"],
        frame_scores=frames,
        summary=ReportSummary(
            max_reba_score=8,
            pct_frames_high_risk=0.3,
            recommendations=["Reducir la flexión del tronco", "Añadir soporte para los brazos"]
        )
    )
```

### test_generator.py
- `test_generate_creates_pdf_file`: el archivo PDF existe tras `generate()`
- `test_pdf_is_not_empty`: el PDF tiene más de 1 KB
- `test_generate_with_empty_frames`: reporte con lista vacía de frames no lanza excepción
- `test_charts_embedded`: el HTML generado contiene imágenes base64

## Dependencias

```
# requirements/reports.txt
weasyprint>=60.0
jinja2>=3.1.0
matplotlib>=3.8.0
loguru>=0.7.0
pytest>=8.0.0
```

## Comandos

```bash
cd reports
pip install -r ../requirements/reports.txt
pytest tests/ -v
# Smoke test — genera un PDF de ejemplo:
python -c "
from reports import ReportGenerator
# usar fixture sample_report como objeto real para smoke test
"
```

## Notas

- **WeasyPrint en Windows**: puede requerir `GTK3` runtime. Alternativa: usar `reportlab` si WeasyPrint da problemas en el OS objetivo. Implementar ambas y elegir según disponibilidad con `try/except`.
- **Fuentes**: usar fuentes del sistema (Arial) para evitar dependencias externas.
- **Tamaño del PDF**: mantener por debajo de 5 MB. Comprimir imágenes base64 si es necesario.

## NO HACER

- No importar de `pose_pipeline`, `ergo_engine`, `llm_rules`, `api`, o `advanced_pipeline`
- No procesar video ni calcular ángulos
- No conectar a Ollama
- No modificar archivos en `schemas/`
