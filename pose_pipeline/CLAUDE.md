# Task A — pose_pipeline/

## Objetivo
Implementar el pipeline completo de estimación de postura 3D desde video monocular. Recibe un video MP4 y devuelve una `SkeletonSequence` con coordenadas 3D en metros.

## Dependencia previa
**Requiere que `schemas/` esté completo.** Importa ÚNICAMENTE de `schemas`.

## Archivos a crear

```
pose_pipeline/
├── __init__.py           # Expone: PosePipeline
├── detector.py           # Detección de persona (YOLOv8n ONNX)
├── pose_2d.py            # Estimación 2D (RTMPose-m ONNX)
├── pose_3d.py            # Lifting 2D→3D (MotionBERT Lite)
├── height.py             # Anclaje de altura y recuperación de escala
├── pipeline.py           # PosePipeline — orquesta las 4 etapas
├── model_downloader.py   # Descarga lazy de modelos ONNX
├── conftest.py           # Fixtures pytest (video de prueba sintético)
└── tests/
    ├── test_detector.py
    ├── test_pose_2d.py
    ├── test_pose_3d.py
    ├── test_height.py
    └── test_pipeline.py
```

## Implementación por archivo

### model_downloader.py
Descarga los modelos ONNX al directorio `MODEL_DIR` (de variable de entorno, default `./models`).

```python
MODELS = {
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        "filename": "yolov8n.onnx",
        "sha256": "..."  # verificar integridad
    },
    "rtmpose_m": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "filename": "rtmpose-m.onnx",
    },
    "motionbert_lite": {
        "url": "https://github.com/Walter0807/MotionBERT/releases/...",
        "filename": "motionbert_lite.onnx",
    }
}

def ensure_model(name: str) -> Path:
    """Descarga el modelo si no existe. Retorna el path local."""
```

### detector.py
```python
class PersonDetector:
    """Detecta bounding boxes de personas usando YOLOv8n ONNX."""

    def __init__(self, model_path: str, device: str = "cpu"):
        ...

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """
        Args:
            frame: imagen BGR (H, W, 3)
        Returns:
            lista de BBox [x1, y1, x2, y2, confidence] para todas las personas
        """

    def detect_primary(self, frame: np.ndarray) -> Optional[BBox]:
        """Retorna solo el BBox de la persona con mayor área (operador principal)."""
```

### pose_2d.py
```python
class PoseEstimator2D:
    """Estimación de 17-26 keypoints 2D usando RTMPose-m ONNX."""

    def __init__(self, model_path: str, device: str = "cpu"):
        # Usar onnxruntime.InferenceSession
        # ExecutionProviders: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ...

    def estimate(self, frame: np.ndarray, bbox: BBox) -> dict[str, Keypoint2D]:
        """
        Args:
            frame: imagen BGR completa
            bbox: bounding box de la persona
        Returns:
            dict de nombre_keypoint -> Keypoint2D(x, y, confidence)
            Usar KEYPOINT_NAMES de schemas.skeleton
        """

    def estimate_batch(self, frames: list[np.ndarray], bboxes: list[BBox]) -> list[dict]:
        """Procesa múltiples frames en batch para mayor throughput."""
```

### pose_3d.py
```python
class PoseLifter3D:
    """
    Lifting 2D→3D usando MotionBERT Lite.
    Procesa ventanas de 243 frames de keypoints 2D normalizados.
    """
    WINDOW_SIZE = 243  # frames por ventana (fijo en MotionBERT)
    STRIDE = 121       # solapamiento 50% entre ventanas

    def __init__(self, model_path: str, device: str = "cpu"):
        ...

    def lift(self, keypoints_2d_sequence: list[dict[str, Keypoint2D]]) -> list[dict[str, Keypoint3D]]:
        """
        Args:
            keypoints_2d_sequence: secuencia de N frames de keypoints 2D
        Returns:
            secuencia de N frames de keypoints 3D (coordenadas normalizadas, root-relative)
        Nota: usa ventana deslizante con WINDOW_SIZE y STRIDE.
        Las coordenadas 3D de salida son root-relative (cadera como origen) y normalizadas.
        La conversión a metros la hace HeightAnchor.
        """
```

### height.py
```python
class HeightAnchor:
    """
    Convierte coordenadas 3D root-relative normalizadas a metros reales
    usando la altura de la persona como ancla de escala.
    También soporta estimación automática por proporciones antropométricas.
    """

    def compute_scale_factor(
        self,
        skeleton_3d: dict[str, Keypoint3D],
        person_height_cm: float
    ) -> float:
        """
        Calcula el factor px→metros usando la altura real.
        Método: distancia desde cabeza (nose) hasta punto medio de tobillos.
        """

    def estimate_height_from_skeleton(
        self,
        skeleton_3d: dict[str, Keypoint3D]
    ) -> float:
        """
        Fallback: estima la altura en cm usando proporciones antropométricas.
        Relación cabeza/cuerpo estándar: altura = 7.5 * altura_cabeza.
        Retorna estimación en cm.
        """

    def apply_scale(
        self,
        skeletons: list[dict[str, Keypoint3D]],
        scale_m_per_unit: float
    ) -> list[dict[str, Keypoint3D]]:
        """Aplica el factor de escala a todos los keypoints. Retorna nuevas instancias."""
```

### pipeline.py
```python
from schemas import VideoInput, SkeletonSequence, Skeleton3D

class PosePipeline:
    """
    Orquestador del pipeline completo de estimación de postura.
    Entrada: VideoInput
    Salida: SkeletonSequence
    """

    def __init__(self, device: str = "cpu"):
        self.detector = PersonDetector(ensure_model("yolov8n"), device)
        self.pose_2d = PoseEstimator2D(ensure_model("rtmpose_m"), device)
        self.lifter_3d = PoseLifter3D(ensure_model("motionbert_lite"), device)
        self.height_anchor = HeightAnchor()

    def process(self, video_input: VideoInput) -> SkeletonSequence:
        """
        Pipeline completo:
        1. Leer video con OpenCV (respetar fps_sample_rate)
        2. Detectar persona en cada frame
        3. Estimar keypoints 2D
        4. Acumular secuencia y aplicar MotionBERT Lite (ventana de 243 frames)
        5. Calcular factor de escala (usar person_height_cm)
        6. Retornar SkeletonSequence con coordenadas en metros
        """

    def process_video_path(self, path: str, person_height_cm: float = 170.0) -> SkeletonSequence:
        """Shortcut para llamadas simples sin VideoInput completo."""
```

## Interfaz Pública (`__init__.py`)

```python
from .pipeline import PosePipeline
from .model_downloader import ensure_model

__all__ = ["PosePipeline", "ensure_model"]
```

## Tests

### conftest.py
Crear un video sintético de 5 segundos a 30fps con una figura humana simple (rectángulos) para tests sin GPU:
```python
@pytest.fixture
def synthetic_video_path(tmp_path):
    # Generar video con cv2.VideoWriter con figura estática
    ...

@pytest.fixture
def mock_keypoints_2d():
    # Secuencia de 243 frames de keypoints sintéticos
    ...
```

### test_pipeline.py
- `test_process_returns_skeleton_sequence`: con video sintético, debe retornar `SkeletonSequence`
- `test_skeleton_has_correct_keypoints`: todos los keypoints de KEYPOINT_NAMES presentes
- `test_coordinates_in_meters`: con altura 170cm, el rango Z del skeleton debe ser ~[0, 1.7]
- `test_sample_rate`: con fps_sample_rate=2, analizar la mitad de frames

## Dependencias

```
# requirements/pose.txt
onnxruntime>=1.17.0          # CPU inference (sin CUDA)
# onnxruntime-gpu>=1.17.0   # Descomentar para GPU
opencv-python>=4.8.0
numpy>=1.24.0
requests>=2.31.0             # Para model_downloader
tqdm>=4.66.0                 # Barra de progreso al descargar
loguru>=0.7.0
pytest>=8.0.0
```

## Comandos

```bash
cd pose_pipeline
pip install -r ../requirements/pose.txt
pytest tests/ -v
# Smoke test manual:
python -c "from pose_pipeline import PosePipeline; p = PosePipeline(); print('OK')"
```

## Notas de Implementación

- **MotionBERT Lite**: Si el modelo ONNX no está disponible públicamente aún, implementar con PyTorch y exportar a ONNX durante el setup. El repo oficial es https://github.com/Walter0807/MotionBERT
- **Normalización 2D**: RTMPose devuelve coordenadas en píxeles del crop. Normalizar a [-1, 1] antes de pasar a MotionBERT.
- **Ventana deslizante**: Si el video tiene menos de 243 frames, hacer padding con el primer/último frame.
- **Coordenadas de mundo vs. cámara**: En el modo CPU_ONLY, las coordenadas son camera-relative. En GPU_ENHANCED (Task F), serán world-grounded vía GVHMR.

## NO HACER

- No importar de `ergo_engine`, `llm_rules`, `api`, `reports`, o `advanced_pipeline`
- No implementar análisis ergonómico aquí (eso es Task B)
- No conectar a Ollama ni LLMs
- No generar reportes PDF
- No modificar archivos en `schemas/`
