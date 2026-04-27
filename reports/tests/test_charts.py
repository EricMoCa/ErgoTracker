import base64
import pytest
from reports.charts import ChartGenerator
from reports.conftest import make_frame
from schemas import RiskLevel, FrameErgonomicScore


@pytest.fixture
def gen():
    return ChartGenerator()


@pytest.fixture
def frame_scores():
    return [make_frame(i, RiskLevel.HIGH if i > 5 else RiskLevel.MEDIUM) for i in range(10)]


def test_risk_timeline_returns_string(gen, frame_scores):
    result = gen.risk_timeline(frame_scores)
    assert isinstance(result, str)
    assert len(result) > 100


def test_risk_timeline_is_valid_base64(gen, frame_scores):
    result = gen.risk_timeline(frame_scores)
    decoded = base64.b64decode(result)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes


def test_risk_timeline_empty_scores(gen):
    result = gen.risk_timeline([])
    assert isinstance(result, str)
    assert len(result) > 0


def test_risk_distribution_pie_returns_string(gen, frame_scores):
    result = gen.risk_distribution_pie(frame_scores)
    assert isinstance(result, str)
    assert len(result) > 100


def test_risk_distribution_pie_valid_base64(gen, frame_scores):
    result = gen.risk_distribution_pie(frame_scores)
    decoded = base64.b64decode(result)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


def test_risk_distribution_pie_empty(gen):
    result = gen.risk_distribution_pie([])
    assert isinstance(result, str)


def test_joint_angles_timeline_returns_string(gen, frame_scores):
    result = gen.joint_angles_timeline(frame_scores, joint_name="trunk_flexion")
    assert isinstance(result, str)


def test_fig_to_base64(gen):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    result = gen._fig_to_base64(fig)
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
