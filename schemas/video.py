from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProcessingMode(str, Enum):
    CPU_ONLY = "cpu_only"
    GPU_ENHANCED = "gpu_enhanced"


class VideoInput(BaseModel):
    path: str
    person_height_cm: float = Field(default=170.0, ge=100.0, le=250.0)
    fps_sample_rate: int = Field(default=1, ge=1, le=30)
    processing_mode: ProcessingMode = ProcessingMode.CPU_ONLY
    ergo_methods: list[str] = Field(default=["REBA"])
    rules_profile_path: Optional[str] = None
    # Advanced pipeline options (only relevant when processing_mode=GPU_ENHANCED)
    preferred_engine: str = "auto"          # auto | gvhmr | wham | tram | humanmm
    requires_gait_analysis: bool = False    # enables GVHMR+contact or WHAM
    has_multiple_shots: bool = False        # enables HumanMM
    camera_motion_high: bool = False        # hints router toward GVHMR
