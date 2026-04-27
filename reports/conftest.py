import pytest
from datetime import datetime
from schemas import (
    AnalysisReport, FrameErgonomicScore, REBAScore, RULAScore,
    RiskLevel, ReportSummary, ErgonomicRule, RuleViolation,
)


def make_reba(total: int = 5, risk: RiskLevel = RiskLevel.MEDIUM) -> REBAScore:
    return REBAScore(
        total=total, risk_level=risk,
        neck_score=2, trunk_score=2, legs_score=1,
        upper_arm_score=2, lower_arm_score=1, wrist_score=1,
        group_a=4, group_b=3,
    )


def make_frame(idx: int, risk: RiskLevel = RiskLevel.MEDIUM) -> FrameErgonomicScore:
    return FrameErgonomicScore(
        frame_idx=idx,
        timestamp_s=float(idx),
        reba=make_reba(8 if risk == RiskLevel.HIGH else 5, risk),
        overall_risk=risk,
    )


@pytest.fixture
def sample_report() -> AnalysisReport:
    frames = []
    for i in range(10):
        risk = RiskLevel.HIGH if i > 6 else RiskLevel.MEDIUM
        frames.append(make_frame(i, risk))

    return AnalysisReport(
        id="test-001",
        created_at=datetime(2026, 1, 1, 12, 0),
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
            recommendations=["Reducir la flexión del tronco", "Añadir soporte para los brazos"],
        ),
    )


@pytest.fixture
def empty_report() -> AnalysisReport:
    return AnalysisReport(
        id="test-empty",
        created_at=datetime(2026, 1, 1, 12, 0),
        video_path="/videos/test.mp4",
        duration_s=0.0,
        total_frames=0,
        analyzed_frames=0,
        person_height_cm=170.0,
        methods_used=["REBA"],
        frame_scores=[],
        summary=ReportSummary(),
    )
