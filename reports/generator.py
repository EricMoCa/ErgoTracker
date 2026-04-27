from pathlib import Path
from collections import defaultdict
from loguru import logger

import jinja2

from schemas import AnalysisReport, RuleViolation, RiskLevel, FrameErgonomicScore
from .charts import ChartGenerator

# Detectar backend PDF disponible
_PDF_BACKEND: str = "none"

try:
    import weasyprint  # type: ignore
    _PDF_BACKEND = "weasyprint"
    logger.debug("Backend PDF: WeasyPrint")
except Exception:
    try:
        import reportlab  # type: ignore
        _PDF_BACKEND = "reportlab"
        logger.debug("Backend PDF: reportlab (fallback)")
    except Exception:
        logger.warning("No hay backend PDF disponible (ni WeasyPrint ni reportlab). PDF no se generará.")


def _render_pdf_weasyprint(html_content: str, output_path: Path) -> None:
    """Genera el PDF usando WeasyPrint."""
    import weasyprint
    weasyprint.HTML(string=html_content).write_pdf(str(output_path))


def _render_pdf_reportlab(html_content: str, output_path: Path) -> None:
    """
    Genera el PDF usando reportlab.
    Parsea el HTML simplificado y extrae texto plano + imágenes base64.
    """
    import re
    import base64
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    styles = getSampleStyleSheet()

    # Estilos personalizados
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=16,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
    )
    center_style = ParagraphStyle(
        "CenterStyle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # --- Extraer texto plano del HTML para estructura básica ---
    # Portada
    story.append(Paragraph("Reporte de Análisis Ergonómico", title_style))
    story.append(Spacer(1, 0.5 * cm))

    # Extraer contenido relevante del HTML con regex simples
    def strip_tags(text: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        # Decode HTML entities
        clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
        return clean

    # Extraer secciones h2
    sections = re.split(r"<h2[^>]*>", html_content)

    for i, section in enumerate(sections):
        if i == 0:
            # Portada — extraer algunos datos
            cover_text = strip_tags(section)
            if cover_text:
                for line in cover_text.split("\n"):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(line, center_style))
            story.append(Spacer(1, 1 * cm))
            continue

        # Separar título de sección del contenido
        parts = section.split("</h2>", 1)
        if len(parts) == 2:
            sec_title = strip_tags(parts[0])
            sec_content = parts[1]
        else:
            sec_title = ""
            sec_content = section

        if sec_title:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2E75B6")))
            story.append(Paragraph(sec_title, heading_style))

        # Extraer imágenes base64
        img_matches = re.findall(r'src="data:image/png;base64,([^"]+)"', sec_content)
        for b64_data in img_matches:
            try:
                img_bytes = base64.b64decode(b64_data)
                img_buf = io.BytesIO(img_bytes)
                img = Image(img_buf, width=16 * cm, height=8 * cm)
                story.append(img)
                story.append(Spacer(1, 0.3 * cm))
            except Exception as e:
                logger.warning(f"No se pudo embeber imagen: {e}")

        # Extraer tablas
        table_matches = re.findall(r"<table[^>]*>(.*?)</table>", sec_content, re.DOTALL)
        for table_html in table_matches:
            rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
            table_data = []
            is_first_row = True
            for row_html in rows_html:
                cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL)
                row = [strip_tags(c) for c in cells]
                if row:
                    table_data.append(row)

            if table_data:
                col_count = max(len(r) for r in table_data)
                col_width = (16 * cm) / col_count if col_count > 0 else 8 * cm
                col_widths = [col_width] * col_count

                tbl = Table(table_data, colWidths=col_widths)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F7FF")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 5),
                    ("WORDWRAP", (0, 0), (-1, -1), True),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 0.3 * cm))

        # Extraer párrafos de recomendaciones (•)
        para_matches = re.findall(r"<p[^>]*>(.*?)</p>", sec_content, re.DOTALL)
        for para_html in para_matches:
            text = strip_tags(para_html)
            if text:
                story.append(Paragraph(text, normal_style))

    doc.build(story)


class ReportGenerator:
    """
    Genera reportes PDF profesionales desde un AnalysisReport.
    Pipeline: AnalysisReport → Jinja2 HTML → WeasyPrint (o reportlab) → PDF
    """

    def __init__(self) -> None:
        self.charts = ChartGenerator()
        template_dir = Path(__file__).parent / "templates"
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        # Filtro Jinja2 personalizado para basename de path
        self.env.filters["basename"] = lambda p: Path(str(p)).name

    def generate(self, report: AnalysisReport, output_path: str) -> str:
        """
        Genera el reporte PDF completo.

        Args:
            report: AnalysisReport con todos los datos del análisis.
            output_path: ruta de destino del PDF (ej: "reporte_2026.pdf").

        Returns:
            path absoluto del PDF generado.
        """
        logger.info(f"Generando reporte PDF para {report.video_path}...")

        # 1. Generar gráficas
        charts_data = {
            "risk_timeline": self.charts.risk_timeline(report.frame_scores),
            "risk_pie": self.charts.risk_distribution_pie(report.frame_scores),
        }

        # 2. Recopilar violaciones únicas más frecuentes
        violations = self._get_top_violations(report.frame_scores, top_n=10)

        # 3. Calcular max_risk_level para la portada
        risk_order = [
            RiskLevel.NEGLIGIBLE, RiskLevel.LOW, RiskLevel.MEDIUM,
            RiskLevel.HIGH, RiskLevel.VERY_HIGH,
        ]
        max_risk = RiskLevel.NEGLIGIBLE
        if report.frame_scores:
            for fs in report.frame_scores:
                if risk_order.index(fs.overall_risk) > risk_order.index(max_risk):
                    max_risk = fs.overall_risk

        # Añadir max_risk_level al summary si no existe (campo dinámico)
        # ReportSummary no tiene max_risk_level nativo, lo pasamos al template directamente
        summary_with_max = report.summary.model_copy()
        # Usamos un wrapper para exponer max_risk_level al template
        class SummaryWrapper:
            def __init__(self, summary, max_risk_level: RiskLevel) -> None:
                self._s = summary
                self.max_risk_level = max_risk_level

            def __getattr__(self, name: str):  # type: ignore[override]
                return getattr(self._s, name)

        summary_wrapped = SummaryWrapper(report.summary, max_risk)

        # Crear wrapper del report con summary aumentado
        class ReportWrapper:
            def __init__(self, report: AnalysisReport, summary_wrapped: SummaryWrapper) -> None:
                self._r = report
                self.summary = summary_wrapped

            def __getattr__(self, name: str):  # type: ignore[override]
                return getattr(self._r, name)

        report_wrapped = ReportWrapper(report, summary_wrapped)

        # 4. Renderizar HTML con Jinja2
        template = self.env.get_template("report_template.html")
        html_content = template.render(
            report=report_wrapped,
            charts=charts_data,
            violations=violations,
        )

        # 5. Convertir HTML → PDF
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if _PDF_BACKEND == "weasyprint":
            _render_pdf_weasyprint(html_content, output_path_obj)
        elif _PDF_BACKEND == "reportlab":
            _render_pdf_reportlab(html_content, output_path_obj)
        else:
            # Sin backend: guardar el HTML como fallback de emergencia
            html_path = output_path_obj.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            logger.warning(f"Sin backend PDF. HTML guardado en: {html_path}")
            return str(html_path)

        size_kb = output_path_obj.stat().st_size // 1024
        logger.success(f"Reporte generado: {output_path_obj} ({size_kb} KB)")
        return str(output_path_obj.resolve())

    def _get_top_violations(
        self,
        frame_scores: list[FrameErgonomicScore],
        top_n: int = 10,
    ) -> list[RuleViolation]:
        """
        Extrae las N violaciones más frecuentes de toda la secuencia.
        Agrupa por rule.id y retorna la peor instancia de cada regla.
        """
        violations_by_rule: dict[str, list[RuleViolation]] = defaultdict(list)
        for frame in frame_scores:
            for v in frame.rule_violations:
                violations_by_rule[v.rule.id].append(v)

        # Ordenar por frecuencia y retornar la peor instancia de cada regla
        sorted_rules = sorted(
            violations_by_rule.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        return [
            max(violations, key=lambda v: v.measured_angle)
            for _, violations in sorted_rules[:top_n]
        ]
