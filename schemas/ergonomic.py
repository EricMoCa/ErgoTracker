from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class RiskLevel(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class ErgonomicRule(BaseModel):
    id: str
    description: str
    joint: str
    condition: str
    risk_level: RiskLevel
    action: str
    source: str


class RuleViolation(BaseModel):
    rule: ErgonomicRule
    measured_angle: float
    threshold: float


class REBAScore(BaseModel):
    total: int = Field(ge=1, le=15)
    risk_level: RiskLevel
    neck_score: int
    trunk_score: int
    legs_score: int
    upper_arm_score: int
    lower_arm_score: int
    wrist_score: int
    group_a: int
    group_b: int
    activity_score: int = 0


class RULAScore(BaseModel):
    total: int = Field(ge=1, le=7)
    risk_level: RiskLevel
    upper_arm_score: int
    lower_arm_score: int
    wrist_score: int
    wrist_twist_score: int
    neck_score: int
    trunk_score: int
    legs_score: int
    group_a: int
    group_b: int
    muscle_use_score: int = 0
    force_load_score: int = 0


class OWASCode(BaseModel):
    back_code: int = Field(ge=1, le=4)
    arms_code: int = Field(ge=1, le=3)
    legs_code: int = Field(ge=1, le=7)
    load_code: int = Field(ge=1, le=3)
    action_level: int = Field(ge=1, le=4)
    risk_level: RiskLevel


class FrameErgonomicScore(BaseModel):
    frame_idx: int
    timestamp_s: float
    reba: Optional[REBAScore] = None
    rula: Optional[RULAScore] = None
    owas: Optional[OWASCode] = None
    rule_violations: list[RuleViolation] = []
    overall_risk: RiskLevel = RiskLevel.NEGLIGIBLE


class ReportSummary(BaseModel):
    max_reba_score: Optional[int] = None
    max_rula_score: Optional[int] = None
    pct_frames_high_risk: float = 0.0
    pct_frames_medium_risk: float = 0.0
    most_violated_rules: list[str] = []
    recommendations: list[str] = []


class AnalysisReport(BaseModel):
    id: str
    created_at: datetime
    video_path: str
    duration_s: float
    total_frames: int
    analyzed_frames: int
    person_height_cm: float
    methods_used: list[str]
    frame_scores: list[FrameErgonomicScore]
    summary: ReportSummary
