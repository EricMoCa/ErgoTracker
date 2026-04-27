"""
Tests para RuleCache.
"""
import json
from pathlib import Path

import pytest

from llm_rules.rule_cache import RuleCache
from schemas import ErgonomicRule, RiskLevel


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rules() -> list[ErgonomicRule]:
    return [
        ErgonomicRule(
            id="R-001",
            description="Flexión de tronco máx 45°",
            joint="trunk_flexion",
            condition="angle > 45",
            risk_level=RiskLevel.HIGH,
            action="Reducir ángulo",
            source="PDF:test.pdf",
        ),
        ErgonomicRule(
            id="R-002",
            description="Elevación del hombro máx 60°",
            joint="shoulder_elevation_right",
            condition="angle > 60",
            risk_level=RiskLevel.MEDIUM,
            action="Bajar el plano de trabajo",
            source="PDF:test.pdf",
        ),
    ]


@pytest.fixture
def cache(tmp_path) -> RuleCache:
    """RuleCache apuntando a un directorio temporal."""
    return RuleCache(cache_dir=tmp_path / "test_cache")


# ---------------------------------------------------------------------------
# Tests de get / set
# ---------------------------------------------------------------------------


def test_get_returns_none_when_not_cached(cache):
    """get retorna None para un hash que no existe en caché."""
    result = cache.get("nonexistent_hash_abc123")
    assert result is None


def test_set_and_get_roundtrip(cache, sample_rules):
    """Las reglas guardadas con set se recuperan correctamente con get."""
    pdf_hash = "abc123def456" * 4 + "ab"  # 64 chars
    cache.set(pdf_hash, sample_rules)
    retrieved = cache.get(pdf_hash)
    assert retrieved is not None
    assert len(retrieved) == len(sample_rules)


def test_set_creates_cache_file(cache, sample_rules):
    """set crea un archivo .json en el directorio de caché."""
    pdf_hash = "a" * 64
    cache.set(pdf_hash, sample_rules)
    expected_file = cache.cache_dir / f"{pdf_hash}.json"
    assert expected_file.exists()


def test_get_returns_ergonomic_rule_instances(cache, sample_rules):
    """get retorna instancias de ErgonomicRule, no dicts crudos."""
    pdf_hash = "b" * 64
    cache.set(pdf_hash, sample_rules)
    retrieved = cache.get(pdf_hash)
    assert all(isinstance(r, ErgonomicRule) for r in retrieved)


def test_set_preserves_rule_fields(cache, sample_rules):
    """Los campos de las reglas se preservan tras la serialización/deserialización."""
    pdf_hash = "c" * 64
    cache.set(pdf_hash, sample_rules)
    retrieved = cache.get(pdf_hash)

    original = sample_rules[0]
    restored = retrieved[0]
    assert restored.id == original.id
    assert restored.description == original.description
    assert restored.joint == original.joint
    assert restored.condition == original.condition
    assert restored.risk_level == original.risk_level
    assert restored.action == original.action
    assert restored.source == original.source


def test_cache_dir_created_automatically(tmp_path):
    """RuleCache crea el directorio de caché si no existe."""
    new_dir = tmp_path / "new_cache_dir"
    assert not new_dir.exists()
    cache = RuleCache(cache_dir=new_dir)
    assert new_dir.exists()


def test_set_empty_rules(cache):
    """set acepta lista vacía de reglas."""
    pdf_hash = "d" * 64
    cache.set(pdf_hash, [])
    retrieved = cache.get(pdf_hash)
    assert retrieved == []


# ---------------------------------------------------------------------------
# Tests de save_profile
# ---------------------------------------------------------------------------


def test_save_profile_creates_file(cache, sample_rules, tmp_path):
    """save_profile crea el archivo JSON en la ruta especificada."""
    output_path = str(tmp_path / "my_profile.json")
    cache.save_profile("ergonomia_industrial", sample_rules, output_path)
    assert Path(output_path).exists()


def test_save_profile_json_structure(cache, sample_rules, tmp_path):
    """save_profile produce JSON con estructura {profile_name, rules}."""
    output_path = str(tmp_path / "profile.json")
    cache.save_profile("test_profile", sample_rules, output_path)
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert "profile_name" in data
    assert "rules" in data
    assert data["profile_name"] == "test_profile"
    assert len(data["rules"]) == len(sample_rules)


def test_save_profile_rules_are_dicts(cache, sample_rules, tmp_path):
    """Las reglas en save_profile son dicts serializables."""
    output_path = str(tmp_path / "profile2.json")
    cache.save_profile("test", sample_rules, output_path)
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    for rule_dict in data["rules"]:
        assert isinstance(rule_dict, dict)
        assert "id" in rule_dict
        assert "joint" in rule_dict
        assert "risk_level" in rule_dict


def test_save_profile_empty_rules(cache, tmp_path):
    """save_profile acepta lista vacía de reglas."""
    output_path = str(tmp_path / "empty_profile.json")
    cache.save_profile("empty", [], output_path)
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert data["rules"] == []
