import matplotlib
matplotlib.use("Agg")  # Sin display (headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import base64
from typing import Optional

from schemas import AnalysisReport, RiskLevel, FrameErgonomicScore

# Colores por nivel de riesgo
RISK_COLORS = {
    RiskLevel.NEGLIGIBLE: "#22C55E",   # verde
    RiskLevel.LOW:        "#86EFAC",   # verde claro
    RiskLevel.MEDIUM:     "#FCD34D",   # amarillo
    RiskLevel.HIGH:       "#F97316",   # naranja
    RiskLevel.VERY_HIGH:  "#EF4444",   # rojo
}

# Orden de niveles de riesgo para comparación
RISK_ORDER = [
    RiskLevel.NEGLIGIBLE,
    RiskLevel.LOW,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.VERY_HIGH,
]


class ChartGenerator:
    """Genera gráficas para el reporte. Retorna imágenes como base64 PNG."""

    def risk_timeline(self, frame_scores: list[FrameErgonomicScore], fps: float = 25.0) -> str:
        """
        Gráfica de línea: eje X = tiempo (segundos), eje Y = score REBA/RULA.
        Colorear fondo por nivel de riesgo.
        Retorna imagen PNG en base64 para embeber en HTML.
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        if not frame_scores:
            ax.set_title("Evolución Temporal del Riesgo")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Score REBA")
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            return self._fig_to_base64(fig)

        times = [fs.timestamp_s for fs in frame_scores]
        reba_scores = [
            fs.reba.total if fs.reba is not None else None
            for fs in frame_scores
        ]
        risk_levels = [fs.overall_risk for fs in frame_scores]

        # Dibujar franjas de fondo por nivel de riesgo
        if len(times) > 1:
            for i in range(len(times) - 1):
                ax.axvspan(
                    times[i], times[i + 1],
                    facecolor=RISK_COLORS[risk_levels[i]],
                    alpha=0.25,
                    linewidth=0,
                )
        elif len(times) == 1:
            ax.axvspan(
                times[0], times[0] + 1,
                facecolor=RISK_COLORS[risk_levels[0]],
                alpha=0.25,
                linewidth=0,
            )

        # Línea principal de scores REBA
        valid_times = [t for t, s in zip(times, reba_scores) if s is not None]
        valid_scores = [s for s in reba_scores if s is not None]

        if valid_times:
            ax.plot(valid_times, valid_scores, color="#1F4E79", linewidth=2,
                    marker="o", markersize=4, label="Score REBA")
            ax.set_ylim(0, 16)
        else:
            ax.set_ylim(0, 16)

        # Líneas de umbral
        ax.axhline(y=4, color="#FCD34D", linestyle="--", linewidth=1, alpha=0.7, label="Umbral Medio (4)")
        ax.axhline(y=7, color="#F97316", linestyle="--", linewidth=1, alpha=0.7, label="Umbral Alto (7)")
        ax.axhline(y=11, color="#EF4444", linestyle="--", linewidth=1, alpha=0.7, label="Umbral Muy Alto (11)")

        # Leyenda de niveles de riesgo
        legend_patches = [
            mpatches.Patch(facecolor=RISK_COLORS[r], alpha=0.5, label=r.value)
            for r in RISK_ORDER
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

        ax.set_title("Evolución Temporal del Riesgo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Score REBA")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._fig_to_base64(fig)

    def risk_distribution_pie(self, frame_scores: list[FrameErgonomicScore]) -> str:
        """
        Gráfica de torta: distribución de frames por nivel de riesgo.
        Retorna imagen PNG en base64.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        if not frame_scores:
            ax.set_title("Distribución de Riesgo")
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            return self._fig_to_base64(fig)

        # Contar frames por nivel de riesgo
        counts: dict[RiskLevel, int] = {r: 0 for r in RISK_ORDER}
        for fs in frame_scores:
            counts[fs.overall_risk] += 1

        # Filtrar niveles sin datos
        labels = []
        sizes = []
        colors = []
        for r in RISK_ORDER:
            if counts[r] > 0:
                labels.append(f"{r.value}\n({counts[r]} frames)")
                sizes.append(counts[r])
                colors.append(RISK_COLORS[r])

        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.85,
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
        else:
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        ax.set_title("Distribución de Riesgo por Frames")
        fig.tight_layout()

        return self._fig_to_base64(fig)

    def joint_angles_timeline(
        self,
        frame_scores: list[FrameErgonomicScore],
        joint_name: str = "trunk_flexion"
    ) -> str:
        """
        Gráfica de ángulo articular específico a lo largo del tiempo.
        Línea horizontal de umbral de riesgo si aplica.
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        if not frame_scores:
            ax.set_title(f"Ángulo Articular: {joint_name}")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Ángulo (°)")
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            return self._fig_to_base64(fig)

        # Extraer tiempos y ángulos de las violaciones de reglas relacionadas al joint
        times = []
        angles = []

        for fs in frame_scores:
            angle_val = None
            for v in fs.rule_violations:
                if joint_name in v.rule.joint:
                    angle_val = v.measured_angle
                    break
            times.append(fs.timestamp_s)
            angles.append(angle_val)

        # Si no hay datos de violaciones, intentar extraer desde REBA
        has_data = any(a is not None for a in angles)

        if not has_data and joint_name == "trunk_flexion":
            # Usar trunk_score de REBA como proxy
            for i, fs in enumerate(frame_scores):
                if fs.reba is not None:
                    angles[i] = float(fs.reba.trunk_score * 15)  # escalar a grados aprox.

        valid_times = [t for t, a in zip(times, angles) if a is not None]
        valid_angles = [a for a in angles if a is not None]

        if valid_times:
            ax.plot(valid_times, valid_angles, color="#2E75B6", linewidth=2,
                    marker="o", markersize=4, label=joint_name)

            # Umbral de riesgo medio (ej: 60° para flexión de tronco)
            threshold = 60.0
            ax.axhline(y=threshold, color="#F97316", linestyle="--",
                       linewidth=1.5, alpha=0.8, label=f"Umbral de riesgo ({threshold}°)")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "Sin datos para esta articulación",
                    transform=ax.transAxes, ha="center", va="center", color="gray")

        ax.set_title(f"Ángulo Articular: {joint_name.replace('_', ' ').title()}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Ángulo (°)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convierte una figura matplotlib a string base64 PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()
