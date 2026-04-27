# Task 0 — schemas/ (FUNDACIÓN)

## Objetivo
Definir todos los contratos de datos compartidos del proyecto ErgoTracker como modelos Pydantic v2. Este módulo es la **única fuente de verdad** para los tipos de datos que fluyen entre módulos. Debe completarse ANTES que cualquier otro módulo.

## Tu tarea
Implementar todos los modelos de datos en los archivos indicados. No hay lógica de negocio aquí, solo definiciones de tipos con validación Pydantic.

## Archivos a crear

```
schemas/
├── __init__.py          # Exporta todos los modelos públicos
├── video.py             # Tipos de input/configuración
├── skeleton.py          # Tipos del esqueleto 3D
├── angles.py            # Tipos de ángulos articulares
├── ergonomic.py         # Tipos de análisis ergonómico y reportes
└── tests/
    └── test_schemas.py  # Validación de instanciación y serialización
```

## Implementación detallada

### schemas/video.py
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ProcessingMode(str, Enum):
    CPU_ONLY = "cpu_only"           # RTMPose ONNX + MotionBERT Lite (CPU)
    GPU_ENHANCED = "gpu_enhanced"   # GVHMR + STRIDE (GPU libre post-LLM)

class VideoInput(BaseModel):
    path: str
    person_height_cm: float = Field(default=170.0, ge=100.0, le=250.0)
    fps_sample_rate: int = Field(default=1, ge=1, le=30)  # analizar 1 de cada N frames
    processing_mode: ProcessingMode = ProcessingMode.CPU_ONLY
    ergo_methods: list[str] = Field(default=["REBA"])  # ["REBA","RULA","OWAS","LLM"]
    rules_profile_path: Optional[str] = None  # JSON de reglas LLM pre-extraídas
```

### schemas/skeleton.py
```python
from pydantic import BaseModel, Field
from typing import Optional

class Keypoint3D(BaseModel):
    x: float   # metros, sistema de coordenadas mundo
    y: float
    z: float
    confidence: float = Field(ge=0.0, le=1.0)
    occluded: bool = False

# Nombres de keypoints (subset de RTMPose wholebody relevante para ergonomía)
KEYPOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "mid_hip", "mid_shoulder",
    "right_eye", "left_eye",
]

class Skeleton3D(BaseModel):
    frame_idx: int
    timestamp_s: float
    keypoints: dict[str, Keypoint3D]  # nombre -> Keypoint3D
    scale_px_to_m: float              # factor de escala aplicado
    person_height_cm: float
    coordinate_system: str = "world"  # "world" | "camera"

class SkeletonSequence(BaseModel):
    video_path: str
    fps: float
    total_frames: int
    frames: list[Skeleton3D]
```

### schemas/angles.py
```python
from pydantic import BaseModel, Field
from typing import Optional

class JointAngles(BaseModel):
    """Ángulos articulares en grados, calculados desde Skeleton3D."""
    frame_idx: int
    timestamp_s: float

    # Tronco
    trunk_flexion: Optional[float] = None       # 0° = erecto, + = flexión anterior
    trunk_lateral_bending: Optional[float] = None
    trunk_rotation: Optional[float] = None

    # Cuello
    neck_flexion: Optional[float] = None
    neck_lateral_bending: Optional[float] = None

    # Hombros
    shoulder_elevation_left: Optional[float] = None   # grados sobre horizontal
    shoulder_elevation_right: Optional[float] = None
    shoulder_abduction_left: Optional[float] = None
    shoulder_abduction_right: Optional[float] = None

    # Codos
    elbow_flexion_left: Optional[float] = None   # 0° = extendido
    elbow_flexion_right: Optional[float] = None

    # Muñecas
    wrist_flexion_left: Optional[float] = None
    wrist_flexion_right: Optional[float] = None
    wrist_deviation_left: Optional[float] = None   # desviación ulnar/radial
    wrist_deviation_right: Optional[float] = None

    # Cadera y piernas
    hip_flexion_left: Optional[float] = None
    hip_flexion_right: Optional[float] = None
    knee_flexion_left: Optional[float] = None
    knee_flexion_right: Optional[float] = None
```

### schemas/ergonomic.py
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime

class RiskLevel(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"   # Sin riesgo
    LOW = "LOW"                 # Riesgo bajo — monitorear
    MEDIUM = "MEDIUM"           # Riesgo medio — investigar y cambiar
    HIGH = "HIGH"               # Riesgo alto — cambiar pronto
    VERY_HIGH = "VERY_HIGH"     # Riesgo muy alto — cambiar inmediatamente

class ErgonomicRule(BaseModel):
    """Regla ergonómica extraída de PDF por LLM o definida manualmente."""
    id: str
    description: str
    joint: str                  # nombre del ángulo en JointAngles (e.g. "trunk_flexion")
    condition: str              # expresión evaluable: "angle > 60"
    risk_level: RiskLevel
    action: str                 # acción correctiva recomendada
    source: str                 # "REBA_standard" | "PDF:nombre_archivo.pdf:p14"

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
    group_a: int    # postura cuello+tronco+piernas
    group_b: int    # postura brazos+muñeca
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
    pct_frames_high_risk: float = 0.0       # % frames con riesgo HIGH o VERY_HIGH
    pct_frames_medium_risk: float = 0.0
    most_violated_rules: list[str] = []     # IDs de reglas más frecuentes
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
```

### schemas/__init__.py
Exportar todos los modelos públicos:
```python
from .video import VideoInput, ProcessingMode
from .skeleton import Keypoint3D, Skeleton3D, SkeletonSequence, KEYPOINT_NAMES
from .angles import JointAngles
from .ergonomic import (
    RiskLevel, ErgonomicRule, RuleViolation,
    REBAScore, RULAScore, OWASCode,
    FrameErgonomicScore, ReportSummary, AnalysisReport
)
```

## Tests a implementar (schemas/tests/test_schemas.py)

1. Instanciar cada modelo con datos válidos → no debe lanzar excepción
2. Instanciar con datos inválidos → debe lanzar `ValidationError`
3. Serializar a JSON y deserializar → debe ser idempotente
4. Verificar que `AnalysisReport.frame_scores` acepta lista vacía
5. Verificar rangos: `REBAScore.total` fuera de [1,15] debe fallar

## Dependencias

```
# requirements/base.txt
pydantic>=2.5.0
```

## Comandos

```bash
cd schemas
pip install pydantic>=2.5.0
pytest tests/ -v
```

## NO HACER
- No añadir lógica de negocio (cálculos, inferencia, I/O)
- No importar de otros módulos del proyecto
- No cambiar los nombres de campos una vez publicado (breaking change para todos)
- No usar tipos que no sean JSON-serializable nativamente
