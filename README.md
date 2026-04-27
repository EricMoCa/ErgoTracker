# ErgoTracker

Plataforma offline de análisis ergonómico post-proceso. Extrae posturas 3D desde video monocular y evalúa riesgos muscoesqueléticos mediante REBA, RULA, OWAS y reglas personalizadas generadas por LLM. Genera reportes PDF profesionales. Corre completamente en local en laptops empresariales (i7, 4 GB VRAM).

---

## Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Uso desde la UI](#uso-desde-la-ui)
- [Uso desde la API](#uso-desde-la-api)
- [Uso Programático (Python)](#uso-programático-python)
- [Extracción de Reglas LLM](#extracción-de-reglas-llm)
- [Tests](#tests)
- [Arquitectura](#arquitectura)
- [Variables de Entorno](#variables-de-entorno)
- [Hardware Objetivo](#hardware-objetivo)

---

## Características

| Función | Detalle |
|---|---|
| Estimación de pose 3D | YOLOv8n + RTMPose-m + MotionBERT Lite (ONNX, CPU/GPU) |
| Análisis ergonómico | REBA, RULA, OWAS — estándares publicados |
| Reglas personalizadas | Gemma 3 4B (Ollama) extrae reglas desde PDFs de normativa |
| Reporte PDF | Gráficas temporales, distribución de riesgo, recomendaciones |
| API REST | FastAPI con análisis en background y polling de estado |
| Dashboard web | React + Vite con gráficas en tiempo real |
| 100 % offline | Sin dependencias de nube una vez instalado |

---

## Requisitos

| Componente | Versión mínima |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ (solo para la UI) |
| RAM | 8 GB (16 GB recomendado) |
| GPU (opcional) | 4 GB VRAM NVIDIA para modos avanzados |
| Ollama | Solo para extracción de reglas LLM |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/EricMoCa/ErgoTracker.git
cd ErgoTracker
```

### 2. Ejecutar el script de setup

**Windows:**
```bat
scripts\setup.bat
```

**Linux / macOS:**
```bash
bash scripts/setup.sh
```

El script crea un entorno virtual `.venv`, instala todas las dependencias Python e instala las dependencias del frontend (`npm install`).

### 3. (Opcional) Ollama para extracción de reglas LLM

Solo necesario si vas a extraer reglas desde PDFs de normativa.

```bash
# Instalar Ollama: https://ollama.com/download
ollama pull gemma3:4b
ollama serve
```

### 4. (Opcional) Modelos ONNX manuales

Los modelos se descargan automáticamente en `./models/` al primer uso. Si la red está restringida, colócalos manualmente:

```
models/
├── yolov8n.onnx
├── rtmpose-m.onnx
└── motionbert_lite.onnx
```

Para cambiar el directorio: `MODEL_DIR=./mis_modelos` en el archivo `.env`.

---

## Inicio Rápido

**Windows:**
```bat
scripts\start.bat
```

**Linux / macOS:**
```bash
bash scripts/start.sh
```

Esto arranca la API y la UI en paralelo:

| Servicio | URL |
|---|---|
| Dashboard (UI) | http://localhost:5173 |
| API REST | http://localhost:8000 |
| Documentación interactiva | http://localhost:8000/docs |

Para detener: `Ctrl+C` (Linux/macOS) o cualquier tecla (Windows).

---

## Uso desde la UI

1. Abre **http://localhost:5173** en el navegador
2. En la pestaña **Análisis de Video**:
   - Arrastra o selecciona un video MP4/AVI/MOV
   - Introduce la altura de la persona (cm)
   - Marca los métodos a aplicar (REBA, RULA, OWAS)
   - Pulsa **Iniciar Análisis**
3. El monitor muestra el progreso en tiempo real (polling cada 2s)
4. Al completarse aparecen:
   - Resumen: duración, frames analizados, % riesgo alto
   - Gráfica temporal de scores REBA/RULA
   - Gráfica de distribución de riesgo
   - Botón para descargar el PDF
5. En la pestaña **Extracción de Reglas LLM**:
   - Sube un PDF con normativa ergonómica
   - Indica un nombre de perfil (ej: `iso_11226`)
   - Pulsa **Extraer Reglas** — usa Gemma 3 vía Ollama

---

## Uso desde la API

### Iniciar un análisis

```bash
curl -X POST http://localhost:8000/analyze/ \
  -F "video=@/ruta/al/video.mp4" \
  -F "person_height_cm=175.0" \
  -F "ergo_methods=REBA,RULA"
```

Respuesta:
```json
{ "job_id": "abc-123", "status": "pending" }
```

### Consultar estado (repetir hasta `completed`)

```bash
curl http://localhost:8000/analyze/abc-123
```

```json
{
  "job_id": "abc-123",
  "status": "completed",
  "result": { "id": "abc-123", "summary": { ... }, ... }
}
```

### Descargar el PDF

```bash
curl -O http://localhost:8000/reports/abc-123/download
```

### Extraer reglas desde PDF

```bash
curl -X POST http://localhost:8000/rules/extract \
  -F "pdf=@normativa_iso_11228.pdf" \
  -F "profile_name=iso_11228"
```

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `POST` | `/analyze/` | Iniciar análisis (background) |
| `GET` | `/analyze/{job_id}` | Consultar estado del análisis |
| `POST` | `/rules/extract` | Extraer reglas desde PDF (LLM) |
| `GET` | `/rules/{job_id}` | Estado de extracción |
| `GET` | `/reports/{id}/download` | Descargar PDF |

---

## Uso Programático (Python)

```python
from schemas import VideoInput, ProcessingMode
from pose_pipeline import PosePipeline
from ergo_engine import ErgoEngine
from reports import ReportGenerator
import uuid
from datetime import datetime

video_input = VideoInput(
    path="video.mp4",
    person_height_cm=175.0,
    ergo_methods=["REBA", "RULA", "OWAS"],
    fps_sample_rate=1,
    processing_mode=ProcessingMode.CPU_ONLY,
)

skeleton_seq = PosePipeline(device="cpu").process(video_input)
frame_scores = ErgoEngine(methods=["REBA", "RULA", "OWAS"]).analyze(skeleton_seq)

from schemas import AnalysisReport, ReportSummary
report = AnalysisReport(
    id=str(uuid.uuid4()),
    created_at=datetime.now(),
    video_path=video_input.path,
    duration_s=len(skeleton_seq.frames) / skeleton_seq.fps,
    total_frames=skeleton_seq.total_frames,
    analyzed_frames=len(frame_scores),
    person_height_cm=video_input.person_height_cm,
    methods_used=video_input.ergo_methods,
    frame_scores=frame_scores,
    summary=ReportSummary(pct_frames_high_risk=0.0),
)
ReportGenerator().generate(report, "output/reporte.pdf")
```

---

## Extracción de Reglas LLM

Requiere Ollama corriendo con `gemma3:4b`.

```python
from llm_rules import RuleExtractor

extractor = RuleExtractor()
rules = extractor.extract("normativa_iso_11228.pdf", profile_name="iso_11228")
# Genera: iso_11228_rules.json
# Usar con: ergo_methods=["LLM"], rules_profile_path="iso_11228_rules.json"
```

La caché de extracción está en `.ergo_cache/` (clave: SHA-256 del PDF). Borrar para forzar re-extracción.

**Nota de GPU:** La extracción LLM y el análisis de video no deben correr simultáneamente si comparten GPU. El sistema gestiona esto automáticamente a través de `GPUManager`.

---

## Tests

```bash
# Todos los tests (excluye GPU e integración) — ~8s
scripts\run_tests.bat           # Windows
bash scripts/run_tests.sh       # Linux/macOS

# Módulo específico
scripts\run_tests.bat ergo_engine
bash scripts/run_tests.sh pose_pipeline

# Con tests de integración (requiere Ollama)
pytest llm_rules/ -v -m integration

# Con tests de GPU (requiere CUDA)
pytest advanced_pipeline/ -v -m gpu

# Test específico
pytest ergo_engine/tests/test_reba.py::test_reba_standing_upright -v
```

Estado actual: **243 tests, 0 fallos**.

---

## Arquitectura

```
schemas           ← fuente de verdad de todos los tipos (Pydantic v2)
pose_pipeline     ← video → SkeletonSequence 3D (ONNX / sintético)
ergo_engine       ← SkeletonSequence → list[FrameErgonomicScore]
llm_rules         ← PDF → ErgonomicRule[] via Gemma 3 (Ollama)
reports           ← AnalysisReport → PDF (reportlab / WeasyPrint)
advanced_pipeline ← motores GPU avanzados: GVHMR, WHAM, TRAM, HumanMM
api               ← FastAPI REST — único orquestador
frontend/         ← React + Vite dashboard
scripts/          ← setup / start / run_tests (Windows + Linux)
```

**Regla de dependencias:** Ningún módulo importa de otro excepto de `schemas` y sus dependencias explícitas. `api/` es el único punto que une todo.

### Flujo de datos

```
VideoInput
  → PosePipeline       → SkeletonSequence
  → ErgoEngine         → list[FrameErgonomicScore]
  → ReportGenerator    → PDF
```

### Motores de estimación avanzados (GPU)

| Motor | Cuándo usar | Disponibilidad |
|---|---|---|
| MotionBERT Lite | CPU fallback, siempre disponible | Descarga automática |
| GVHMR | GPU, cámara móvil, >5 min | `pip install -e GVHMR/` |
| TRAM | GPU ≥4 GB, si GVHMR falla | `pip install -e tram/` |
| WHAM | Análisis de marcha, <2 min | Solo Linux |
| HumanMM | Video multi-plano | `pip install -e HumanMM-code/` |

---

## Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
OLLAMA_HOST=http://localhost:11434
MODEL_DIR=./models
LOG_LEVEL=INFO
MAX_VIDEO_SIZE_MB=500
DEFAULT_PERSON_HEIGHT_CM=170.0
REPORTS_OUTPUT_DIR=./output/reports
```

---

## Hardware Objetivo

- **CPU:** Intel i7 (8+ núcleos)
- **RAM:** 16 GB
- **GPU:** NVIDIA 4 GB VRAM (base) / 6-8 GB para TRAM o GVHMR
- **OS:** Windows 10/11 (Linux compatible excepto WHAM)

---

## Licencia

Uso interno / investigación. Ver NDA si aplica.
