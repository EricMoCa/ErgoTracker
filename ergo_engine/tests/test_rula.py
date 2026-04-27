import pytest
from schemas import RiskLevel
from ergo_engine.rula import RULAAnalyzer
from ergo_engine.conftest import make_upright_angles, make_high_risk_angles


@pytest.fixture
def analyzer():
    return RULAAnalyzer()


def test_rula_standing_upright(analyzer):
    angles = make_upright_angles()
    result = analyzer.analyze(angles)
    assert 1 <= result.total <= 7
    assert result.risk_level in [RiskLevel.NEGLIGIBLE, RiskLevel.LOW]


def test_rula_high_risk(analyzer):
    angles = make_high_risk_angles()
    result = analyzer.analyze(angles)
    assert result.total >= 4
    assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.VERY_HIGH]


def test_rula_score_range(analyzer):
    angles = make_upright_angles()
    result = analyzer.analyze(angles)
    assert 1 <= result.total <= 7


def test_rula_risk_levels(analyzer):
    assert analyzer._risk_level_from_score(1) == RiskLevel.NEGLIGIBLE
    assert analyzer._risk_level_from_score(2) == RiskLevel.NEGLIGIBLE
    assert analyzer._risk_level_from_score(3) == RiskLevel.LOW
    assert analyzer._risk_level_from_score(4) == RiskLevel.LOW
    assert analyzer._risk_level_from_score(5) == RiskLevel.MEDIUM
    assert analyzer._risk_level_from_score(6) == RiskLevel.MEDIUM
    assert analyzer._risk_level_from_score(7) == RiskLevel.VERY_HIGH


def test_rula_all_fields_present(analyzer):
    result = analyzer.analyze(make_upright_angles())
    assert result.upper_arm_score >= 1
    assert result.lower_arm_score >= 1
    assert result.wrist_score >= 1
    assert result.neck_score >= 1
    assert result.trunk_score >= 1
    assert result.legs_score >= 1
    assert result.group_a >= 1
    assert result.group_b >= 1
