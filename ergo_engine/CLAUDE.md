# Task B — ergo_engine/

## Objetivo
Implementar el motor de análisis ergonómico. Recibe una `SkeletonSequence` (o lista de `JointAngles`) y devuelve una lista de `FrameErgonomicScore` con puntuaciones REBA, RULA, OWAS y violaciones de reglas personalizadas.

## Dependencia previa
**Requiere que `schemas/` esté completo.** Importa ÚNICAMENTE de `schemas`. NO depende de `pose_pipeline`.

## Archivos a crear

```
ergo_engine/
├── __init__.py               # Expone: ErgoEngine, JointAngleCalculator
├── angles.py                 # Calculadora de ángulos desde Skeleton3D
├── reba.py                   # Implementación REBA completa
├── rula.py                   # Implementación RULA completa
├── owas.py                   # Implementación OWAS completa
├── llm_rule_analyzer.py      # Evaluador de reglas ErgonomicRule
├── engine.py                 # ErgoEngine — orquestador
├── conftest.py               # Fixtures con skeletons y ángulos sintéticos
└── tests/
    ├── test_angles.py
    ├── test_reba.py
    ├── test_rula.py
    ├── test_owas.py
    ├── test_llm_rule_analyzer.py
    └── test_engine.py
```

## Implementación por archivo

### angles.py
```python
from schemas import Skeleton3D, JointAngles
import numpy as np

class JointAngleCalculator:
    """
    Calcula ángulos articulares en grados desde keypoints 3D.
    Fórmula: angle = arccos(dot(v1,v2) / (|v1|*|v2|)) * (180/π)
    donde v1 = punto_proximal - articulacion, v2 = punto_distal - articulacion
    """

    def calculate(self, skeleton: Skeleton3D) -> JointAngles:
        """Calcula todos los ángulos posibles desde un Skeleton3D."""

    def _angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Ángulo en el punto b, formado por los segmentos b-a y b-c.
        Retorna grados [0, 180].
        """

    def _trunk_flexion(self, skeleton: Skeleton3D) -> Optional[float]:
        """
        Ángulo entre la línea vertical y el segmento mid_hip→mid_shoulder.
        0° = erecto, 90° = horizontal.
        """

    def _neck_flexion(self, skeleton: Skeleton3D) -> Optional[float]:
        """Ángulo entre mid_shoulder→neck y neck→nose."""

    # Implementar todos los ángulos definidos en schemas/angles.py
```

### reba.py
Implementar el algoritmo REBA completo según la publicación original (Hignett & McAtamney, 2000).

```python
from schemas import JointAngles, REBAScore, RiskLevel

class REBAAnalyzer:
    """
    Rapid Entire Body Assessment (REBA).
    Referencia: Hignett & McAtamney (2000). Applied Ergonomics, 31(2), 201-205.
    """

    def analyze(self, angles: JointAngles) -> REBAScore:
        """
        Calcula la puntuación REBA completa.
        Retorna REBAScore con todos los sub-scores y nivel de riesgo.
        """

    def _score_neck(self, neck_flexion: float) -> int:
        """
        1: 0-20° flexión
        2: >20° flexión o extensión
        +1: si hay torsión o inclinación lateral
        """

    def _score_trunk(self, trunk_flexion: float, lateral: float = 0) -> int:
        """
        1: 0° (erecto)
        2: 0-20° flexión o 0-20° extensión
        3: 20-60° flexión o >20° extensión
        4: >60° flexión
        +1: si hay torsión o inclinación lateral
        """

    def _score_legs(self, knee_flexion: float) -> int:
        """
        1: soporte bilateral, caminando o sentado
        2: soporte unilateral o postura inestable
        +1: 30-60° flexión de rodilla
        +2: >60° flexión de rodilla
        """

    def _score_upper_arm(self, shoulder_elevation: float) -> int:
        """
        1: 20° extensión a 20° flexión
        2: >20° extensión o 20-45° flexión
        3: 45-90° flexión
        4: >90° flexión
        +1: si hombro elevado o brazo abducido
        -1: si hay apoyo del brazo
        """

    def _score_lower_arm(self, elbow_flexion: float) -> int:
        """
        1: 60-100° flexión
        2: <60° o >100° flexión
        """

    def _score_wrist(self, wrist_flexion: float, wrist_deviation: float) -> int:
        """
        1: 0-15° flexión/extensión
        2: >15° flexión/extensión
        +1: si hay desviación o torsión
        """

    def _risk_level_from_score(self, score: int) -> RiskLevel:
        """
        1:       NEGLIGIBLE
        2-3:     LOW
        4-7:     MEDIUM
        8-10:    HIGH
        11-15:   VERY_HIGH
        """
```

### rula.py
Implementar el algoritmo RULA completo (McAtamney & Corlett, 1993).

```python
from schemas import JointAngles, RULAScore, RiskLevel

class RULAAnalyzer:
    """
    Rapid Upper Limb Assessment (RULA).
    Referencia: McAtamney & Corlett (1993). Applied Ergonomics, 24(2), 91-99.
    """

    def analyze(self, angles: JointAngles) -> RULAScore:
        """Calcula la puntuación RULA completa."""

    # Scores Grupo A: upper_arm, lower_arm, wrist, wrist_twist
    def _score_upper_arm_rula(self, elevation: float) -> int: ...
    def _score_lower_arm_rula(self, flexion: float) -> int: ...
    def _score_wrist_rula(self, flexion: float) -> int: ...

    # Scores Grupo B: neck, trunk, legs
    def _score_neck_rula(self, flexion: float) -> int: ...
    def _score_trunk_rula(self, flexion: float) -> int: ...

    # Tablas A y B del estándar RULA (implementar como dict o numpy array)
    TABLE_A: dict  # [upper_arm][lower_arm][wrist] → score
    TABLE_B: dict  # [neck][trunk][legs] → score
    TABLE_C: dict  # [score_a][score_b] → score_final

    def _risk_level_from_score(self, score: int) -> RiskLevel:
        """
        1-2: NEGLIGIBLE
        3-4: LOW
        5-6: MEDIUM
        7:   VERY_HIGH
        """
```

### owas.py
```python
from schemas import JointAngles, OWASCode, RiskLevel

class OWASAnalyzer:
    """
    Ovako Working Posture Analysis System (OWAS).
    Referencia: Karhu et al. (1977). Journal of Occupational Health.
    """

    def analyze(self, angles: JointAngles, load_kg: float = 0) -> OWASCode:
        """
        Clasifica postura en 4 categorías + nivel de acción.
        load_kg: peso cargado (0=sin carga, <10=ligero, <20=moderado, ≥20=pesado)
        """

    def _classify_back(self, trunk_flexion: float, trunk_rotation: float) -> int:
        """1=erecto, 2=inclinado, 3=rotado, 4=inclinado+rotado"""

    def _classify_arms(self, shoulder_elev_l: float, shoulder_elev_r: float) -> int:
        """1=ambos abajo, 2=uno arriba, 3=ambos arriba"""

    def _classify_legs(self, knee_flexion_l: float, knee_flexion_r: float) -> int:
        """1-7 según posición: sentado, de pie, caminando, etc."""

    def _classify_load(self, load_kg: float) -> int:
        """1=<10kg, 2=10-20kg, 3=>20kg"""

    # Tabla de acción OWAS (4x4 array): [back][arms] → action_level
    ACTION_TABLE: list[list[int]]

    def _risk_level_from_action(self, action_level: int) -> RiskLevel:
        """1=NEGLIGIBLE, 2=LOW, 3=HIGH, 4=VERY_HIGH"""
```

### llm_rule_analyzer.py
```python
from schemas import JointAngles, ErgonomicRule, RuleViolation
import operator

class LLMRuleAnalyzer:
    """
    Evalúa reglas ergonómicas definidas en ErgonomicRule contra los ángulos medidos.
    Las reglas son generadas por llm_rules/ y cargadas desde un JSON.
    """

    # Operadores soportados en la condición: "angle > 60", "angle < 30", etc.
    OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}

    def load_rules(self, rules_json_path: str) -> None:
        """Carga reglas desde archivo JSON generado por llm_rules/."""

    def evaluate(self, angles: JointAngles) -> list[RuleViolation]:
        """
        Evalúa todas las reglas cargadas contra los ángulos del frame.
        Para cada regla, obtiene el valor del ángulo de JointAngles usando getattr(angles, rule.joint).
        Evalúa la condición y retorna las violaciones.
        """

    def _parse_condition(self, condition: str) -> tuple[str, float]:
        """
        Parsea "angle > 60" → (">", 60.0)
        La variable siempre es "angle" (reemplazar por el valor real).
        """
```

### engine.py
```python
from schemas import SkeletonSequence, JointAngles, FrameErgonomicScore, AnalysisReport, RiskLevel
from .angles import JointAngleCalculator
from .reba import REBAAnalyzer
from .rula import RULAAnalyzer
from .owas import OWASAnalyzer
from .llm_rule_analyzer import LLMRuleAnalyzer

class ErgoEngine:
    """
    Motor de análisis ergonómico. Orquesta todos los analizadores.
    Entrada: SkeletonSequence
    Salida: list[FrameErgonomicScore]
    """

    def __init__(
        self,
        methods: list[str] = ["REBA"],    # ["REBA","RULA","OWAS","LLM"]
        rules_json_path: Optional[str] = None,
        load_kg: float = 0.0
    ):
        self.angle_calculator = JointAngleCalculator()
        self.analyzers = {}
        if "REBA" in methods:
            self.analyzers["REBA"] = REBAAnalyzer()
        if "RULA" in methods:
            self.analyzers["RULA"] = RULAAnalyzer()
        if "OWAS" in methods:
            self.analyzers["OWAS"] = OWASAnalyzer()
        if "LLM" in methods and rules_json_path:
            self.llm_analyzer = LLMRuleAnalyzer()
            self.llm_analyzer.load_rules(rules_json_path)

    def analyze(self, skeleton_sequence: SkeletonSequence) -> list[FrameErgonomicScore]:
        """Analiza cada frame del skeleton_sequence y retorna scores."""

    def _compute_overall_risk(self, frame_score: FrameErgonomicScore) -> RiskLevel:
        """El riesgo global es el máximo entre todos los métodos del frame."""
```

## Interfaz Pública (`__init__.py`)
```python
from .engine import ErgoEngine
from .angles import JointAngleCalculator

__all__ = ["ErgoEngine", "JointAngleCalculator"]
```

## Tests Clave

### test_reba.py — Casos de referencia documentados
```python
def test_reba_standing_upright():
    """Postura completamente erguida → REBA score 1-2 (NEGLIGIBLE/LOW)"""
    angles = JointAngles(frame_idx=0, timestamp_s=0.0,
        trunk_flexion=0.0, neck_flexion=0.0,
        knee_flexion_left=0.0, knee_flexion_right=0.0,
        shoulder_elevation_left=10.0, shoulder_elevation_right=10.0,
        elbow_flexion_left=80.0, elbow_flexion_right=80.0)
    result = REBAAnalyzer().analyze(angles)
    assert result.total <= 3
    assert result.risk_level in [RiskLevel.NEGLIGIBLE, RiskLevel.LOW]

def test_reba_severe_flexion():
    """Tronco >60°, cuello >20° → REBA score alto (HIGH/VERY_HIGH)"""
    ...
```

### test_angles.py
- Verificar que ángulo entre dos vectores ortogonales = 90°
- Verificar que postura erguida sintética produce trunk_flexion ≈ 0°

## Dependencias

```
# requirements/ergo.txt
numpy>=1.24.0
loguru>=0.7.0
pytest>=8.0.0
```

## Comandos

```bash
cd ergo_engine
pip install -r ../requirements/ergo.txt
pytest tests/ -v --tb=short
```

## Tablas REBA/RULA de Referencia

Consultar las publicaciones originales o implementaciones validadas:
- REBA: https://www.reba.co.uk/ (tablas A y B)
- RULA: tablas en el artículo original de McAtamney & Corlett (1993)

Las tablas deben estar hardcoded como diccionarios Python con los valores exactos del estándar.

## NO HACER

- No importar de `pose_pipeline`, `llm_rules`, `api`, `reports`, o `advanced_pipeline`
- No cargar modelos de ML (todo es geometría y lookup tables)
- No acceder a archivos de video
- No conectar a Ollama o cualquier servicio externo
- No modificar archivos en `schemas/`
