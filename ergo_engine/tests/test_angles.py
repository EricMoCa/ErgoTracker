import numpy as np
import pytest
from ergo_engine.angles import JointAngleCalculator
from ergo_engine.conftest import make_skeleton


@pytest.fixture
def calc():
    return JointAngleCalculator()


def test_angle_between_orthogonal(calc):
    """Ángulo entre vectores ortogonales = 90°."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(calc._angle_between(a, b, c) - 90.0) < 1e-6


def test_angle_between_same_direction(calc):
    """Vectores en la misma dirección = 0°."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])
    assert abs(calc._angle_between(a, b, c) - 0.0) < 1e-6


def test_angle_between_opposite(calc):
    """Vectores opuestos = 180°."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([-1.0, 0.0, 0.0])
    assert abs(calc._angle_between(a, b, c) - 180.0) < 1e-6


def test_upright_trunk_flexion_near_zero(calc):
    """Postura erecta → trunk_flexion ≈ 0°."""
    skeleton = make_skeleton(upright=True)
    angles = calc.calculate(skeleton)
    assert angles.trunk_flexion is not None
    assert angles.trunk_flexion < 15.0


def test_bent_trunk_flexion_high(calc):
    """Postura inclinada → trunk_flexion > 45°."""
    skeleton = make_skeleton(upright=False)
    angles = calc.calculate(skeleton)
    assert angles.trunk_flexion is not None
    assert angles.trunk_flexion > 30.0


def test_calculate_returns_joint_angles(calc):
    skeleton = make_skeleton(upright=True)
    angles = calc.calculate(skeleton)
    assert angles.frame_idx == skeleton.frame_idx
    assert angles.timestamp_s == skeleton.timestamp_s


def test_elbow_flexion_upright(calc):
    """En postura erecta la flexión de codo debe ser calculable."""
    skeleton = make_skeleton(upright=True)
    angles = calc.calculate(skeleton)
    assert angles.elbow_flexion_left is not None
    assert angles.elbow_flexion_right is not None


def test_knee_flexion_upright_low(calc):
    """En postura erecta la flexión de rodilla debe ser baja."""
    skeleton = make_skeleton(upright=True)
    angles = calc.calculate(skeleton)
    assert angles.knee_flexion_left is not None
    assert angles.knee_flexion_left < 20.0
