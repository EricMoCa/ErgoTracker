import json
import pytest
from schemas import RiskLevel, SkeletonSequence
from ergo_engine import ErgoEngine
from ergo_engine.conftest import make_skeleton


@pytest.fixture
def sequence():
    frames = [make_skeleton(i, upright=True) for i in range(3)]
    return SkeletonSequence(video_path="/v.mp4", fps=25.0, total_frames=75, frames=frames)


@pytest.fixture
def high_risk_sequence():
    frames = [make_skeleton(i, upright=False) for i in range(3)]
    return SkeletonSequence(video_path="/v.mp4", fps=25.0, total_frames=75, frames=frames)


def test_engine_reba_only(sequence):
    engine = ErgoEngine(methods=["REBA"])
    scores = engine.analyze(sequence)
    assert len(scores) == 3
    for s in scores:
        assert s.reba is not None
        assert s.rula is None
        assert s.owas is None


def test_engine_rula_only(sequence):
    engine = ErgoEngine(methods=["RULA"])
    scores = engine.analyze(sequence)
    for s in scores:
        assert s.rula is not None
        assert s.reba is None


def test_engine_owas_only(sequence):
    engine = ErgoEngine(methods=["OWAS"])
    scores = engine.analyze(sequence)
    for s in scores:
        assert s.owas is not None
        assert s.reba is None


def test_engine_all_methods(sequence):
    engine = ErgoEngine(methods=["REBA", "RULA", "OWAS"])
    scores = engine.analyze(sequence)
    for s in scores:
        assert s.reba is not None
        assert s.rula is not None
        assert s.owas is not None


def test_engine_frame_count(sequence):
    engine = ErgoEngine()
    scores = engine.analyze(sequence)
    assert len(scores) == len(sequence.frames)


def test_engine_frame_indices(sequence):
    engine = ErgoEngine()
    scores = engine.analyze(sequence)
    for i, score in enumerate(scores):
        assert score.frame_idx == i


def test_engine_overall_risk_not_none(sequence):
    engine = ErgoEngine(methods=["REBA"])
    scores = engine.analyze(sequence)
    for s in scores:
        assert s.overall_risk is not None
        assert isinstance(s.overall_risk, RiskLevel)


def test_engine_high_risk_detected(high_risk_sequence):
    engine = ErgoEngine(methods=["REBA"])
    scores = engine.analyze(high_risk_sequence)
    max_risk = max(scores, key=lambda s: ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"].index(s.overall_risk))
    assert max_risk.overall_risk in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.VERY_HIGH]


def test_engine_with_llm_rules(tmp_path, sequence):
    rules = [{
        "id": "R-001", "description": "Test", "joint": "trunk_flexion",
        "condition": "angle > 5", "risk_level": "MEDIUM",
        "action": "reduce", "source": "test",
    }]
    f = tmp_path / "rules.json"
    f.write_text(json.dumps(rules))
    engine = ErgoEngine(methods=["REBA", "LLM"], rules_json_path=str(f))
    scores = engine.analyze(sequence)
    assert len(scores) == 3


def test_engine_empty_sequence():
    engine = ErgoEngine()
    empty = SkeletonSequence(video_path="/v.mp4", fps=25.0, total_frames=0, frames=[])
    scores = engine.analyze(empty)
    assert scores == []
