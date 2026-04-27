import pytest
from datetime import datetime
from pydantic import ValidationError

from schemas import (
    VideoInput, ProcessingMode,
    Keypoint3D, Skeleton3D, SkeletonSequence, KEYPOINT_NAMES,
    JointAngles,
    RiskLevel, ErgonomicRule, RuleViolation,
    REBAScore, RULAScore, OWASCode,
    FrameErgonomicScore, ReportSummary, AnalysisReport,
)


# --- Fixtures ---

def make_keypoint(x=0.0, y=0.0, z=0.0, confidence=0.9):
    return Keypoint3D(x=x, y=y, z=z, confidence=confidence)


def make_skeleton(frame_idx=0):
    kps = {name: make_keypoint() for name in KEYPOINT_NAMES}
    return Skeleton3D(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx) / 25.0,
        keypoints=kps,
        scale_px_to_m=0.01,
        person_height_cm=170.0,
    )


def make_rule():
    return ErgonomicRule(
        id="R-001",
        description="Tronco no debe superar 45°",
        joint="trunk_flexion",
        condition="angle > 45",
        risk_level=RiskLevel.HIGH,
        action="Reducir ángulo",
        source="REBA_standard",
    )


def make_reba_score(total=5):
    return REBAScore(
        total=total,
        risk_level=RiskLevel.MEDIUM,
        neck_score=2, trunk_score=2, legs_score=1,
        upper_arm_score=2, lower_arm_score=1, wrist_score=1,
        group_a=4, group_b=3,
    )


# --- VideoInput ---

def test_video_input_defaults():
    v = VideoInput(path="/video.mp4")
    assert v.person_height_cm == 170.0
    assert v.fps_sample_rate == 1
    assert v.processing_mode == ProcessingMode.CPU_ONLY
    assert v.ergo_methods == ["REBA"]


def test_video_input_invalid_height_low():
    with pytest.raises(ValidationError):
        VideoInput(path="/v.mp4", person_height_cm=50.0)


def test_video_input_invalid_height_high():
    with pytest.raises(ValidationError):
        VideoInput(path="/v.mp4", person_height_cm=300.0)


def test_video_input_invalid_fps():
    with pytest.raises(ValidationError):
        VideoInput(path="/v.mp4", fps_sample_rate=0)


# --- Keypoint3D ---

def test_keypoint_valid():
    kp = Keypoint3D(x=1.0, y=2.0, z=3.0, confidence=0.95)
    assert kp.occluded is False


def test_keypoint_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Keypoint3D(x=0, y=0, z=0, confidence=1.5)


# --- Skeleton3D ---

def test_skeleton_instantiation():
    s = make_skeleton()
    assert s.coordinate_system == "world"
    assert len(s.keypoints) == len(KEYPOINT_NAMES)


# --- SkeletonSequence ---

def test_skeleton_sequence_empty_frames():
    seq = SkeletonSequence(video_path="/v.mp4", fps=25.0, total_frames=0, frames=[])
    assert seq.frames == []


def test_skeleton_sequence_with_frames():
    frames = [make_skeleton(i) for i in range(5)]
    seq = SkeletonSequence(video_path="/v.mp4", fps=25.0, total_frames=125, frames=frames)
    assert len(seq.frames) == 5


# --- JointAngles ---

def test_joint_angles_all_optional():
    angles = JointAngles(frame_idx=0, timestamp_s=0.0)
    assert angles.trunk_flexion is None
    assert angles.knee_flexion_left is None


def test_joint_angles_with_values():
    angles = JointAngles(frame_idx=1, timestamp_s=0.04, trunk_flexion=30.0, neck_flexion=15.0)
    assert angles.trunk_flexion == 30.0


# --- REBAScore ---

def test_reba_score_valid():
    r = make_reba_score(total=8)
    assert r.total == 8
    assert r.risk_level == RiskLevel.MEDIUM


def test_reba_score_out_of_range_low():
    with pytest.raises(ValidationError):
        make_reba_score(total=0)


def test_reba_score_out_of_range_high():
    with pytest.raises(ValidationError):
        make_reba_score(total=16)


# --- RULAScore ---

def test_rula_score_valid():
    r = RULAScore(
        total=4, risk_level=RiskLevel.LOW,
        upper_arm_score=2, lower_arm_score=1, wrist_score=1, wrist_twist_score=1,
        neck_score=2, trunk_score=2, legs_score=1,
        group_a=3, group_b=3,
    )
    assert r.total == 4


def test_rula_score_out_of_range():
    with pytest.raises(ValidationError):
        RULAScore(
            total=8, risk_level=RiskLevel.VERY_HIGH,
            upper_arm_score=2, lower_arm_score=1, wrist_score=1, wrist_twist_score=1,
            neck_score=2, trunk_score=2, legs_score=1,
            group_a=3, group_b=3,
        )


# --- OWASCode ---

def test_owas_valid():
    o = OWASCode(back_code=2, arms_code=1, legs_code=3, load_code=1, action_level=2, risk_level=RiskLevel.LOW)
    assert o.action_level == 2


def test_owas_back_code_invalid():
    with pytest.raises(ValidationError):
        OWASCode(back_code=5, arms_code=1, legs_code=1, load_code=1, action_level=1, risk_level=RiskLevel.NEGLIGIBLE)


# --- ErgonomicRule / RuleViolation ---

def test_ergonomic_rule_valid():
    rule = make_rule()
    assert rule.joint == "trunk_flexion"


def test_rule_violation_valid():
    v = RuleViolation(rule=make_rule(), measured_angle=60.0, threshold=45.0)
    assert v.measured_angle == 60.0


# --- FrameErgonomicScore ---

def test_frame_score_defaults():
    fs = FrameErgonomicScore(frame_idx=0, timestamp_s=0.0)
    assert fs.reba is None
    assert fs.rule_violations == []
    assert fs.overall_risk == RiskLevel.NEGLIGIBLE


def test_frame_score_with_reba():
    fs = FrameErgonomicScore(frame_idx=0, timestamp_s=0.0, reba=make_reba_score())
    assert fs.reba.total == 5


# --- AnalysisReport ---

def test_analysis_report_empty_frames():
    report = AnalysisReport(
        id="test-001",
        created_at=datetime.now(),
        video_path="/v.mp4",
        duration_s=10.0,
        total_frames=250,
        analyzed_frames=0,
        person_height_cm=170.0,
        methods_used=["REBA"],
        frame_scores=[],
        summary=ReportSummary(),
    )
    assert report.frame_scores == []


def test_analysis_report_serialization():
    report = AnalysisReport(
        id="test-002",
        created_at=datetime(2026, 1, 1, 12, 0),
        video_path="/v.mp4",
        duration_s=5.0,
        total_frames=125,
        analyzed_frames=5,
        person_height_cm=175.0,
        methods_used=["REBA", "RULA"],
        frame_scores=[FrameErgonomicScore(frame_idx=0, timestamp_s=0.0)],
        summary=ReportSummary(max_reba_score=7, pct_frames_high_risk=0.2),
    )
    data = report.model_dump()
    restored = AnalysisReport(**data)
    assert restored.id == report.id
    assert restored.summary.max_reba_score == 7


def test_keypoint_names_count():
    assert len(KEYPOINT_NAMES) == 18
    assert "neck" in KEYPOINT_NAMES
    assert "mid_hip" in KEYPOINT_NAMES
