# Task F — advanced_pipeline/

## Objetivo
Implementar el pipeline avanzado de estimación de postura 3D con motores intercambiables
world-grounded. Reemplaza opcionalmente a `pose_pipeline` cuando `ProcessingMode.GPU_ENHANCED`
está activo. La interfaz pública es idéntica a `PosePipeline` — siempre retorna `SkeletonSequence`.

## Dependencias previas
**Requiere que `schemas/` y `pose_pipeline/` estén completos.**

## Contexto de GPU
Este módulo SOLO se ejecuta después de que `llm_rules` haya liberado la VRAM.
Verificar que la GPU tiene al menos 3 GB libres antes de cargar modelos.

## Archivos a crear

```
advanced_pipeline/
├── __init__.py
├── pipeline.py              # AdvancedPosePipeline — orquestador principal
├── pipeline_router.py       # Selector automático de motor según video + hardware
├── gvhmr_wrapper.py         # Motor primario world-grounded (SIGGRAPH Asia 2024)
├── wham_wrapper.py          # Motor secundario — contacto pie-suelo (CVPR 2024)
├── tram_wrapper.py          # Motor alternativo — SLAM+VIMO (ECCV 2024)
├── contact_refinement.py    # Post-proceso: detección de contacto pie-suelo
├── stride_refinement.py     # Post-proceso: refinamiento por oclusión (WACV 2025)
├── humanmm_wrapper.py       # Multi-shot video (CVPR 2025)
├── visual_odometry.py       # Estimación de movimiento de cámara (fallback OpenCV)
├── conftest.py
└── tests/
    ├── test_pipeline_router.py
    ├── test_gvhmr_wrapper.py
    ├── test_wham_wrapper.py
    ├── test_contact_refinement.py
    ├── test_stride_refinement.py
    └── test_pipeline.py
```

---

## Contexto de los Modelos

### GVHMR (SIGGRAPH Asia 2024) — MOTOR PRIMARIO
- **Repo**: https://github.com/zju3dv/GVHMR
- **Paper**: "World-Grounded Human Motion Recovery via Gravity-View Coordinates"
- **Qué hace**: Define un sistema Gravity-View (GV) — alineado con gravedad y dirección
  de cámara. Estima poses en GV para cada frame y las compone hacia coordenadas de mundo.
  Usa Transformer + RoPE, procesa secuencias de longitud arbitraria sin sliding window.
- **Ventaja clave sobre WHAM**: Sin drift en secuencias largas (no es RNN autoregresivo).
  7× más rápido en inferencia (0.28s vs 2.0s para 1430 frames en RTX 4090).
  Mejor en benchmarks world-grounded (RICH, EMDB).
- **Actualización (marzo 2025)**: DPVO reemplazado por SimpleVO — instalación en Windows
  sustancialmente más simple.
- **Input**: video RGB
- **Output**: poses SMPL en coordenadas de mundo
- **VRAM requerida**: ~3-4 GB
- **Cuándo usar**: default para cámara móvil, videos largos (>5 min), análisis industrial

### WHAM (CVPR 2024) — MOTOR SECUNDARIO (contacto pie-suelo)
- **Repo**: https://github.com/yohanshin/WHAM
- **Paper**: "WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion"
- **Qué hace**: Pipeline de dos ramas — ViTPose extrae keypoints 2D, DPVO estima movimiento
  de cámara, un RNN autoregresivo predice poses en cámara + trayectoria global.
  Su innovación principal es **contact-aware trajectory recovery**: predice probabilidad
  de contacto pie-suelo y la usa para eliminar foot sliding.
- **Combinación con GVHMR**: El módulo de detección de contacto de WHAM se puede aplicar
  como post-proceso sobre la salida de GVHMR via `contact_refinement.py`. Esto da lo
  mejor de ambos: precisión world-grounded de GVHMR + foot contact de WHAM.
- **Limitaciones**: RNN acumula drift en videos largos. DPVO no compila en Windows
  (issues #100, #87 en GitHub). Requiere PyTorch 1.11 + CUDA 11.3 (versiones antiguas).
- **VRAM requerida**: ~4 GB
- **Cuándo usar**: análisis de marcha/carga, video corto (<2 min), foot contact analysis.
  En v1 implementar como stub. Priorizar `contact_refinement.py` (más portable).

### TRAM (ECCV 2024) — MOTOR ALTERNATIVO
- **Repo**: https://github.com/yufu-wang/tram
- **Paper**: "TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos"
- **Qué hace**: Dos etapas limpias:
  1. DROID-SLAM con doble masking (ignora humanos dinámicos) → trayectoria métrica
  2. VIMO (Video Transformer sobre ViT-H) → movimiento corporal kinematic
  Usa la escena (fondo) para inferir escala métrica automáticamente.
  Reporta 60% reducción de error en trayectoria global.
- **VRAM requerida**: ~4-6 GB (ViT-H es grande)
- **Cuándo usar**: alternativa si GVHMR no disponible, o cuando la escena tiene
  referencias de escala claras (suelo visible, objetos de tamaño conocido).

### STRIDE (WACV 2025) — REFINADOR DE OCLUSIÓN
- **Qué hace**: Test-Time Training sobre un prior de movimiento humano.
  Refina estimaciones con oclusión severa (hasta 100% de frames ocultos).
- **Fallback**: interpolación temporal lineal entre frames válidos adyacentes.

### HumanMM (CVPR 2025) — MULTI-SHOT
- **Repo**: https://github.com/zhangyuhong01/HumanMM-code
- **Cuándo usar**: videos con cortes de cámara / shot transitions.

---

## Implementación por archivo

### pipeline_router.py
```python
from dataclasses import dataclass
from typing import Optional
import torch
from loguru import logger


@dataclass
class VideoProfile:
    """Análisis del video antes de seleccionar el motor."""
    duration_s: float
    has_multiple_shots: bool = False
    camera_motion_score: float = 0.0   # 0=fija, 1=muy móvil
    occlusion_score: float = 0.0       # 0=sin oclusión, 1=oclusión total
    requires_gait_analysis: bool = False


@dataclass
class HardwareProfile:
    vram_gb: float = 0.0
    cuda_available: bool = False

    @classmethod
    def detect(cls) -> "HardwareProfile":
        try:
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                return cls(vram_gb=free_vram, cuda_available=True)
        except Exception:
            pass
        return cls(vram_gb=0.0, cuda_available=False)


class PipelineRouter:
    """
    Selecciona el motor world-grounded más adecuado.

    Orden de prioridad:
    1. HumanMM         → videos multi-plano
    2. GVHMR           → cámara móvil, videos largos (>5 min), default GPU
    3. GVHMR + contact → si requiere análisis de marcha
    4. WHAM            → análisis de marcha en video corto (<2 min)
    5. TRAM            → alternativa si GVHMR no disponible, GPU >= 4GB
    6. MotionBERT Lite → fallback CPU siempre disponible
    """

    def select(
        self,
        video_profile: VideoProfile,
        hardware: Optional[HardwareProfile] = None,
        gvhmr_available: bool = False,
        wham_available: bool = False,
        tram_available: bool = False,
        humanmm_available: bool = False,
    ) -> str:
        """
        Retorna: "humanmm" | "gvhmr" | "gvhmr_with_contact" |
                 "wham" | "tram" | "motionbert_lite"
        """
        hw = hardware or HardwareProfile.detect()

        if video_profile.has_multiple_shots and humanmm_available:
            logger.info("Router → HumanMM (multi-shot video)")
            return "humanmm"

        if hw.cuda_available and hw.vram_gb >= 3.0:
            if gvhmr_available:
                if video_profile.requires_gait_analysis:
                    logger.info("Router → GVHMR + contact_refinement")
                    return "gvhmr_with_contact"
                if video_profile.duration_s > 300 or \
                   video_profile.camera_motion_score > 0.2:
                    logger.info("Router → GVHMR (video largo / cámara móvil)")
                    return "gvhmr"
                logger.info("Router → GVHMR (default GPU)")
                return "gvhmr"

            if wham_available and video_profile.requires_gait_analysis and \
               video_profile.duration_s < 120:
                logger.info("Router → WHAM (gait, video corto)")
                return "wham"

            if tram_available and hw.vram_gb >= 4.0:
                logger.info("Router → TRAM (GVHMR no disponible)")
                return "tram"

        logger.warning("Router → MotionBERT Lite (CPU fallback)")
        return "motionbert_lite"
```

### contact_refinement.py
```python
import numpy as np
from schemas import SkeletonSequence, Skeleton3D
from loguru import logger

FOOT_JOINTS = ["left_ankle", "right_ankle", "left_heel", "right_heel",
               "left_foot_index", "right_foot_index"]


class ContactRefinement:
    """
    Detecta contacto pie-suelo y refina la trayectoria para eliminar foot sliding.

    Extrae la lógica contact-aware de WHAM como post-proceso portable, aplicable
    sobre cualquier SkeletonSequence (GVHMR, TRAM, MotionBERT).

    En v1: detección por velocidad de keypoint.
    En v2: integrar clasificador de contacto de WHAM si está disponible.
    """

    def refine(self, skeleton_seq: SkeletonSequence) -> SkeletonSequence:
        contact_labels = self._detect_contact(skeleton_seq)
        logger.debug(
            f"Contacto: {sum(contact_labels)}/{len(contact_labels)} frames"
        )
        return self._apply_floor_constraint(skeleton_seq, contact_labels)

    def _detect_contact(self, skeleton_seq: SkeletonSequence) -> list[bool]:
        """
        Detecta contacto por velocidad de keypoints de pie.
        Pie en contacto si velocidad < 2cm/frame entre frames consecutivos.
        """
        frames = skeleton_seq.frames
        contact = [False] * len(frames)

        for i in range(1, len(frames)):
            prev, curr = frames[i - 1], frames[i]
            for joint in FOOT_JOINTS:
                if joint not in prev.keypoints or joint not in curr.keypoints:
                    continue
                p_kp = prev.keypoints[joint]
                c_kp = curr.keypoints[joint]
                velocity = np.sqrt(
                    (c_kp.x - p_kp.x) ** 2 +
                    (c_kp.y - p_kp.y) ** 2 +
                    (c_kp.z - p_kp.z) ** 2
                )
                if velocity < 0.02:
                    contact[i] = True
                    break

        return contact

    def _apply_floor_constraint(
        self,
        skeleton_seq: SkeletonSequence,
        contact_labels: list[bool]
    ) -> SkeletonSequence:
        """
        v1: stub seguro — retorna sin modificar.
        v2: fijar posición vertical del pie al plano del suelo en frames con contacto.
        """
        return skeleton_seq

    def get_contact_events(self, skeleton_seq: SkeletonSequence) -> list[dict]:
        """
        Retorna eventos de contacto para análisis ergonómico.
        Útil para distinguir postura estática vs dinámica en REBA/OWAS.
        Returns: [{"start_frame": int, "end_frame": int, "foot": "left"|"right"}]
        """
        ...
```

### gvhmr_wrapper.py
```python
import numpy as np
from schemas import SkeletonSequence
from .visual_odometry import VisualOdometry
from loguru import logger


class GVHMRWrapper:
    """
    Wrapper alrededor de GVHMR — motor primario world-grounded.

    Instalación:
        git clone https://github.com/zju3dv/GVHMR
        pip install -e GVHMR/
        # Desde marzo 2025 usa SimpleVO — instalación simplificada en Windows

    Si GVHMR no está instalado: is_available() retorna False, estimate() retorna None.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self.vo = VisualOdometry(method="opencv_fallback")

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from gvhmr.models.smpl_hmr2 import GVHMR as GVHMRModel
            self._model = GVHMRModel.from_pretrained("zju3dv/GVHMR").to(self.device)
            self._model.eval()
            logger.info("GVHMR cargado en GPU")
        except ImportError:
            logger.warning("GVHMR no instalado. Ver: pip install -e GVHMR/")
            self._model = None

    def is_available(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            from gvhmr.models.smpl_hmr2 import GVHMR
            return True
        except ImportError:
            return False

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """
        Estima poses world-grounded. Retorna None si GVHMR no disponible.

        Pipeline:
        1. SimpleVO estima rotación de cámara por frame
        2. ViTPose extrae keypoints 2D + features
        3. Transformer + RoPE procesa secuencia completa (sin sliding window)
        4. Predicción en sistema GV → composición a coordenadas de mundo
        5. Convertir a SkeletonSequence con coordinate_system="world"
        """
        self._load_model()
        if self._model is None:
            return None
        camera_rotations = self.vo.estimate(video_path)
        # TODO: pipeline completo con GVHMR
        ...
```

### wham_wrapper.py
```python
from schemas import SkeletonSequence
from loguru import logger


class WHAMWrapper:
    """
    Wrapper alrededor de WHAM — motor secundario para foot contact analysis.

    WHAM: contacto pie-suelo nativo, elimina foot sliding.
    Limitaciones: drift en secuencias largas, instalación Windows problemática.

    Alternativa recomendada: usar GVHMRWrapper + ContactRefinement en su lugar.

    Instalación (solo cuando se necesite explícitamente):
        git clone https://github.com/yohanshin/WHAM
        # conda create -n wham python=3.9
        # conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3
        # pip install -v -e third-party/ViTPose
        # DPVO no compila en Windows — usar --estimate_local_only en Windows
        # Descargar SMPL: registro en smpl.is.tue.mpg.de

    v1: stub que retorna (None, None). Priorizar contact_refinement.py.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = self._check_wham()

    def _check_wham(self) -> bool:
        try:
            import wham
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def estimate(
        self,
        video_path: str,
        person_height_cm: float = 170.0
    ) -> tuple[SkeletonSequence | None, list[float] | None]:
        """
        Retorna (SkeletonSequence, contact_probs) o (None, None) si no disponible.
        contact_probs: [float 0-1 por frame] — probabilidad de contacto pie-suelo.
        """
        if not self._available:
            logger.warning("WHAM no instalado. Ver CLAUDE.md.")
            return None, None
        # TODO: implementar cuando WHAM esté instalado
        ...
```

### tram_wrapper.py
```python
from schemas import SkeletonSequence
from loguru import logger


class TRAMWrapper:
    """
    Wrapper alrededor de TRAM (ECCV 2024) — motor alternativo world-grounded.

    Arquitectura:
    1. DROID-SLAM con doble masking → trayectoria métrica de cámara
    2. VIMO (ViT-H video transformer) → movimiento corporal
    60% reducción de error en trayectoria global.

    Instalación:
        git clone https://github.com/yufu-wang/tram
        pip install -e tram/

    v1: stub que retorna None.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = self._check_tram()

    def _check_tram(self) -> bool:
        try:
            import tram
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def estimate(self, video_path: str, person_height_cm: float = 170.0) -> SkeletonSequence | None:
        """Retorna SkeletonSequence con coordinate_system="world" o None."""
        if not self._available:
            logger.warning("TRAM no instalado. Ver: github.com/yufu-wang/tram")
            return None
        # TODO: implementar cuando TRAM esté instalado
        ...
```

### pipeline.py
```python
from schemas import VideoInput, SkeletonSequence
from pose_pipeline import PosePipeline
from .pipeline_router import PipelineRouter, VideoProfile, HardwareProfile
from .gvhmr_wrapper import GVHMRWrapper
from .wham_wrapper import WHAMWrapper
from .tram_wrapper import TRAMWrapper
from .contact_refinement import ContactRefinement
from .stride_refinement import STRIDERefinement
from .humanmm_wrapper import HumanMMWrapper
from loguru import logger
import torch


class AdvancedPosePipeline:
    """
    Pipeline avanzado — drop-in replacement de PosePipeline.
    PipelineRouter selecciona el mejor motor disponible.
    Siempre retorna SkeletonSequence válido.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._verify_gpu()
        self._router = PipelineRouter()
        self._gvhmr = GVHMRWrapper(device)
        self._wham = WHAMWrapper(device)
        self._tram = TRAMWrapper(device)
        self._stride = STRIDERefinement()
        self._humanmm = HumanMMWrapper(device)
        self._contact = ContactRefinement()
        self._fallback = PosePipeline(device="cpu")

    def _verify_gpu(self):
        try:
            if not torch.cuda.is_available():
                logger.warning("GPU no disponible. Usando CPU.")
                self.device = "cpu"
                return
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_vram < 2.5:
                logger.warning(
                    f"VRAM libre: {free_vram:.1f} GB. "
                    "Verificar que Ollama liberó GPU."
                )
        except Exception:
            pass

    def process(self, video_input: VideoInput) -> SkeletonSequence:
        hw = HardwareProfile.detect()
        video_profile = self._build_video_profile(video_input)

        motor = self._router.select(
            video_profile=video_profile,
            hardware=hw,
            gvhmr_available=self._gvhmr.is_available(),
            wham_available=self._wham.is_available(),
            tram_available=self._tram.is_available(),
            humanmm_available=self._humanmm.is_available(),
        )

        logger.info(f"AdvancedPosePipeline → motor: {motor}")
        result = None

        if motor == "humanmm":
            result = self._humanmm.process_multishot(
                video_input.path, video_input.person_height_cm
            )
        elif motor == "gvhmr_with_contact":
            result = self._gvhmr.estimate(video_input.path, video_input.person_height_cm)
            if result:
                result = self._contact.refine(result)
        elif motor == "gvhmr":
            result = self._gvhmr.estimate(video_input.path, video_input.person_height_cm)
        elif motor == "wham":
            seq, _ = self._wham.estimate(video_input.path, video_input.person_height_cm)
            result = seq
        elif motor == "tram":
            result = self._tram.estimate(video_input.path, video_input.person_height_cm)

        if result and video_profile.occlusion_score > 0.2:
            result = self._stride.refine(result)

        if result:
            return result

        logger.warning("Fallback a PosePipeline base (CPU).")
        return self._fallback.process(video_input)

    def _build_video_profile(self, video_input: VideoInput) -> VideoProfile:
        """v1: perfil básico. v2: análisis real del video (duración, motion, oclusión)."""
        return VideoProfile(
            duration_s=0.0,
            camera_motion_score=0.0,
            occlusion_score=0.0,
            has_multiple_shots=False,
            requires_gait_analysis=False,
        )
```

### stride_refinement.py, visual_odometry.py, humanmm_wrapper.py
Ver implementación en los CLAUDE.md de referencia original. Sin cambios.

---

## Tests

### test_pipeline_router.py
- `test_router_selects_gvhmr_for_long_video`: duration=400s → "gvhmr"
- `test_router_selects_humanmm_for_multishot`: has_multiple_shots=True → "humanmm"
- `test_router_selects_gvhmr_with_contact_for_gait`: requires_gait_analysis=True → "gvhmr_with_contact"
- `test_router_fallback_no_gpu`: sin CUDA → "motionbert_lite"
- `test_router_tram_when_gvhmr_unavailable`: GVHMR unavailable, TRAM available + 4GB → "tram"

### test_pipeline.py
- `test_output_always_skeleton_sequence`: siempre retorna SkeletonSequence
- `test_fallback_without_gpu`: sin CUDA → usa PosePipeline base
- `test_contact_refinement_called_for_gait`: gait_analysis activa contact_refinement

### test_contact_refinement.py
- `test_static_foot_detected_as_contact`: keypoint sin movimiento → contact=True
- `test_moving_foot_not_contact`: velocidad alta → contact=False
- `test_refine_returns_skeleton_sequence`: retorna SkeletonSequence válido

### test_stride_refinement.py
- `test_detect_low_confidence_frames`: confidence<0.4 → frame detectado
- `test_interpolation_fills_occluded`: frame ocluido entre dos válidos → interpolado
- `test_no_occlusion_unchanged`: sin oclusión → secuencia idéntica

---

## Dependencias

```
# requirements/advanced.txt
-r base.txt
torch>=2.1.0
opencv-python>=4.8.0
pytest>=8.0.0

# Instalar manualmente:
# GVHMR (primario): git clone https://github.com/zju3dv/GVHMR && pip install -e GVHMR/
# TRAM (alternativo): git clone https://github.com/yufu-wang/tram && pip install -e tram/
# WHAM (foot contact): git clone https://github.com/yohanshin/WHAM
#   → Solo Linux recomendado. En Windows: usar sin DPVO (--estimate_local_only)
# HumanMM: git clone https://github.com/zhangyuhong01/HumanMM-code && pip install -e HumanMM-code/
# STRIDE: consultar paper WACV 2025 para código oficial
```

## NO HACER
- No importar de `ergo_engine`, `llm_rules`, `api`, o `reports`
- No implementar análisis ergonómico
- No conectar a Ollama
- No generar reportes PDF
- No modificar `schemas/` ni `pose_pipeline/`
