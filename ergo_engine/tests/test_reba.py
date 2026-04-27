import pytest
from schemas import RiskLevel
from ergo_engine.reba import REBAAnalyzer
from ergo_engine.conftest import make_upright_angles, make_high_risk_angles, make_angles


@pytest.fixture
def analyzer():
    return REBAAnalyzer()


def test_reba_standing_upright(analyzer):
    """Postura completamente erguida → REBA score bajo."""
    angles = make_upright_angles()
    result = analyzer.analyze(angles)
    assert result.total <= 4
    assert result.risk_level in [RiskLevel.NEGLIGIBLE, RiskLevel.LOW, RiskLevel.MEDIUM]


def test_reba_severe_flexion(analyzer):
    """Tronco >60°, cuello >20°, brazos elevados → REBA score alto."""
    angles = make_high_risk_angles()
    result = analyzer.analyze(angles)
    assert result.total >= 7
    assert result.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.VERY_HIGH]


def test_reba_neck_score_low(analyzer):
    angles = make_angles(neck_flexion=10.0)
    assert analyzer._score_neck(10.0) == 1


def test_reba_neck_score_high(analyzer):
    assert analyzer._score_neck(25.0) == 2


def test_reba_neck_score_lateral(analyzer):
    assert analyzer._score_neck(10.0, lateral=15.0) == 2


def test_reba_trunk_erect(analyzer):
    assert analyzer._score_trunk(0.0) == 1


def test_reba_trunk_mild_flex(analyzer):
    assert analyzer._score_trunk(15.0) == 2


def test_reba_trunk_moderate_flex(analyzer):
    assert analyzer._score_trunk(40.0) == 3


def test_reba_trunk_severe_flex(analyzer):
    assert analyzer._score_trunk(70.0) == 4


def test_reba_trunk_lateral(analyzer):
    score_without = analyzer._score_trunk(10.0, lateral=0.0)
    score_with = analyzer._score_trunk(10.0, lateral=15.0)
    assert score_with == score_without + 1


def test_reba_legs_straight(analyzer):
    assert analyzer._score_legs(0.0) == 1


def test_reba_legs_moderate_flex(analyzer):
    assert analyzer._score_legs(45.0) == 2


def test_reba_legs_severe_flex(analyzer):
    assert analyzer._score_legs(80.0) == 3


def test_reba_upper_arm_low(analyzer):
    assert analyzer._score_upper_arm(10.0) == 1


def test_reba_upper_arm_mid(analyzer):
    assert analyzer._score_upper_arm(35.0) == 2


def test_reba_upper_arm_high(analyzer):
    assert analyzer._score_upper_arm(70.0) == 3


def test_reba_upper_arm_very_high(analyzer):
    assert analyzer._score_upper_arm(100.0) == 4


def test_reba_lower_arm_good_range(analyzer):
    assert analyzer._score_lower_arm(80.0) == 1


def test_reba_lower_arm_bad_range(analyzer):
    assert analyzer._score_lower_arm(40.0) == 2
    assert analyzer._score_lower_arm(110.0) == 2


def test_reba_wrist_neutral(analyzer):
    assert analyzer._score_wrist(10.0) == 1


def test_reba_wrist_flexed(analyzer):
    assert analyzer._score_wrist(20.0) == 2


def test_reba_wrist_deviation(analyzer):
    score_without = analyzer._score_wrist(10.0, 0.0)
    score_with = analyzer._score_wrist(10.0, 15.0)
    assert score_with == score_without + 1


def test_reba_risk_levels(analyzer):
    assert analyzer._risk_level_from_score(1) == RiskLevel.NEGLIGIBLE
    assert analyzer._risk_level_from_score(2) == RiskLevel.LOW
    assert analyzer._risk_level_from_score(3) == RiskLevel.LOW
    assert analyzer._risk_level_from_score(4) == RiskLevel.MEDIUM
    assert analyzer._risk_level_from_score(7) == RiskLevel.MEDIUM
    assert analyzer._risk_level_from_score(8) == RiskLevel.HIGH
    assert analyzer._risk_level_from_score(10) == RiskLevel.HIGH
    assert analyzer._risk_level_from_score(11) == RiskLevel.VERY_HIGH
    assert analyzer._risk_level_from_score(15) == RiskLevel.VERY_HIGH


def test_reba_score_has_all_fields(analyzer):
    result = analyzer.analyze(make_upright_angles())
    assert result.neck_score >= 1
    assert result.trunk_score >= 1
    assert result.legs_score >= 1
    assert result.upper_arm_score >= 1
    assert result.lower_arm_score >= 1
    assert result.wrist_score >= 1
    assert result.group_a >= 1
    assert result.group_b >= 1
