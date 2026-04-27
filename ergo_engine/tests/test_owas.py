import pytest
from schemas import RiskLevel
from ergo_engine.owas import OWASAnalyzer
from ergo_engine.conftest import make_upright_angles, make_high_risk_angles, make_angles


@pytest.fixture
def analyzer():
    return OWASAnalyzer()


def test_owas_upright(analyzer):
    angles = make_upright_angles()
    result = analyzer.analyze(angles)
    assert 1 <= result.back_code <= 4
    assert 1 <= result.arms_code <= 3
    assert 1 <= result.legs_code <= 7
    assert 1 <= result.action_level <= 4


def test_owas_back_erect(analyzer):
    assert analyzer._classify_back(0.0, 0.0) == 1


def test_owas_back_bent(analyzer):
    assert analyzer._classify_back(30.0, 0.0) == 2


def test_owas_back_twisted(analyzer):
    assert analyzer._classify_back(0.0, 20.0) == 3


def test_owas_back_bent_and_twisted(analyzer):
    assert analyzer._classify_back(30.0, 20.0) == 4


def test_owas_arms_both_low(analyzer):
    assert analyzer._classify_arms(10.0, 10.0) == 1


def test_owas_arms_one_high(analyzer):
    assert analyzer._classify_arms(100.0, 10.0) == 2


def test_owas_arms_both_high(analyzer):
    assert analyzer._classify_arms(100.0, 100.0) == 3


def test_owas_load_light(analyzer):
    assert analyzer._classify_load(5.0) == 1


def test_owas_load_medium(analyzer):
    assert analyzer._classify_load(15.0) == 2


def test_owas_load_heavy(analyzer):
    assert analyzer._classify_load(25.0) == 3


def test_owas_risk_levels(analyzer):
    assert analyzer._risk_level_from_action(1) == RiskLevel.NEGLIGIBLE
    assert analyzer._risk_level_from_action(2) == RiskLevel.LOW
    assert analyzer._risk_level_from_action(3) == RiskLevel.HIGH
    assert analyzer._risk_level_from_action(4) == RiskLevel.VERY_HIGH


def test_owas_high_risk_posture(analyzer):
    angles = make_high_risk_angles()
    result = analyzer.analyze(angles)
    assert result.action_level >= 2
