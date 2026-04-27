import json
import pytest
from pathlib import Path
from schemas import RiskLevel
from ergo_engine.llm_rule_analyzer import LLMRuleAnalyzer
from ergo_engine.conftest import make_angles


@pytest.fixture
def rules_file(tmp_path):
    rules = [
        {
            "id": "R-001",
            "description": "Tronco no superar 45°",
            "joint": "trunk_flexion",
            "condition": "angle > 45",
            "risk_level": "HIGH",
            "action": "Reducir ángulo",
            "source": "test",
        },
        {
            "id": "R-002",
            "description": "Cuello no superar 20°",
            "joint": "neck_flexion",
            "condition": "angle > 20",
            "risk_level": "MEDIUM",
            "action": "Ajustar posición",
            "source": "test",
        },
    ]
    f = tmp_path / "rules.json"
    f.write_text(json.dumps(rules), encoding="utf-8")
    return str(f)


@pytest.fixture
def analyzer(rules_file):
    a = LLMRuleAnalyzer()
    a.load_rules(rules_file)
    return a


def test_load_rules(analyzer):
    assert len(analyzer.rules) == 2


def test_no_violation_when_safe(analyzer):
    angles = make_angles(trunk_flexion=30.0, neck_flexion=10.0)
    violations = analyzer.evaluate(angles)
    assert len(violations) == 0


def test_trunk_violation_triggered(analyzer):
    angles = make_angles(trunk_flexion=60.0)
    violations = analyzer.evaluate(angles)
    rule_ids = [v.rule.id for v in violations]
    assert "R-001" in rule_ids


def test_neck_violation_triggered(analyzer):
    angles = make_angles(neck_flexion=25.0)
    violations = analyzer.evaluate(angles)
    rule_ids = [v.rule.id for v in violations]
    assert "R-002" in rule_ids


def test_both_violations(analyzer):
    angles = make_angles(trunk_flexion=60.0, neck_flexion=25.0)
    violations = analyzer.evaluate(angles)
    assert len(violations) == 2


def test_violation_contains_measured_angle(analyzer):
    angles = make_angles(trunk_flexion=60.0)
    violations = analyzer.evaluate(angles)
    assert violations[0].measured_angle == 60.0
    assert violations[0].threshold == 45.0


def test_parse_condition_gt(analyzer):
    op, threshold = analyzer._parse_condition("angle > 60")
    assert op == ">"
    assert threshold == 60.0


def test_parse_condition_gte(analyzer):
    op, threshold = analyzer._parse_condition("angle >= 45")
    assert op == ">="
    assert threshold == 45.0


def test_parse_condition_lt(analyzer):
    op, threshold = analyzer._parse_condition("angle < 30")
    assert op == "<"
    assert threshold == 30.0


def test_missing_joint_skipped(analyzer):
    """Si el ángulo del joint es None, no debe generar violación."""
    angles = make_angles()  # todos None
    violations = analyzer.evaluate(angles)
    assert len(violations) == 0


def test_load_rules_profile_format(tmp_path):
    """Acepta formato de perfil con wrapper {"rules": [...]}."""
    data = {
        "profile_name": "test",
        "rules": [{
            "id": "R-003",
            "description": "Test",
            "joint": "trunk_flexion",
            "condition": "angle > 10",
            "risk_level": "LOW",
            "action": "x",
            "source": "y",
        }]
    }
    f = tmp_path / "profile.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    a = LLMRuleAnalyzer()
    a.load_rules(str(f))
    assert len(a.rules) == 1


def test_missing_file_does_not_crash():
    a = LLMRuleAnalyzer()
    a.load_rules("/nonexistent/path.json")
    assert len(a.rules) == 0
