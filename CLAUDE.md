# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ErgoTracker — Descripción

Plataforma modular de análisis ergonómico post-proceso: extrae posturas 3D desde video monocular y evalúa riesgos muscoesqueléticos (REBA, RULA, OWAS, reglas LLM). Corre completamente offline en laptops empresariales (i7, 4 GB VRAM).

**Estado:** todos los módulos están implementados y con tests completos (243 tests, 0 fallos).

---

## Instalación

### 1. Entorno base (requerido siempre)

```bash
# Python 3.11+ requerido
pip install -r requirements/base.txt
```

### 2. Por funcionalidad

```bash
pip install -r requirements/pose.txt       # pose_pipeline: ONNX, OpenCV
pip install -r requirements/ergo.txt       # ergo_engine:   numpy, loguru
pip install -r requirements/llm.txt        # llm_rules:     pymupdf, requests
pip install -r requirements/reports.txt    # reports:       jinja2, matplotlib, reportlab
pip install -r requirements/advanced.txt   # advanced_pipeline: torch, opencv
pip install -r requirements/api.txt        # api:           fastapi, uvicorn, pydantic-settings
```

### 3. Instalar todo de una vez

```bash
pip install -r requirements/base.txt -r requirements/pose.txt \
            -r requirements/ergo.txt -r requirements/llm.txt \
            -r requirements/reports.txt -r requirements/advanced.txt \
            -r requirements/api.txt
```

### 4. Modelos ONNX (primer uso)

Los modelos se descargan automáticamente en `./models/` al primer uso. Si hay restricciones de red, descargar manualmente y colocar en `MODEL_DIR`:

```
models/
├── yolov8n.onnx            # Detector de persona
├── rtmpose-m.onnx          # Estimación de pose 2D
└── motionbert_lite.onnx    # Lifting 2D→3D
```

Variable de entorno para cambiar el directorio: `MODEL_DIR=./models`

### 5. Ollama + Gemma (solo para extracción LLM de reglas)

```bash
# Instalar Ollama: https://ollama.com/download
ollama pull gemma3:4b
ollama serve   # Puerto 11434 por defecto

# Verificar:
curl http://localhost:11434/api/tags
```

### 6. Motores avanzados (opcionales — GPU Enhanced)

```bash
# GVHMR (motor primario world-grounded):
git clone https://github.com/zju3dv/GVHMR
pip install -e GVHMR/

# TRAM (alternativo):
git clone https://github.com/yufu-wang/tram
pip install -e tram/

# WHAM (foot contact, solo Linux recomendado):
git clone https://github.com/yohanshin/WHAM

# HumanMM (multi-shot):
git clone https://github.com/zhangyuhong01/HumanMM-code
pip install -e HumanMM-code/
```

---

## Ejecución

### API REST (modo principal)

```bash
# Iniciar servidor
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Documentación interactiva:
# http://localhost:8000/docs
```

#### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `POST` | `/analyze/` | Iniciar análisis de video (background) |
| `GET` | `/analyze/{job_id}` | Consultar estado del análisis |
| `POST` | `/rules/extract` | Extraer reglas ergonómicas de PDF (LLM) |
| `GET` | `/rules/{job_id}` | Estado de extracción de reglas |
| `GET` | `/reports/{id}/download` | Descargar PDF del reporte |

#### Ejemplo de uso completo

```bash
# 1. Subir video para análisis
curl -X POST http://localhost:8000/analyze/ \
  -F "video=@/ruta/al/video.mp4" \
  -F "person_height_cm=175.0" \
  -F "ergo_methods=REBA,RULA"

# Respuesta: {"job_id": "abc-123", "status": "pending"}

# 2. Consultar estado (repetir hasta status=completed)
curl http://localhost:8000/analyze/abc-123

# 3. Descargar PDF
curl -O http://localhost:8000/reports/abc-123/download

# 4. (Opcional) Extraer reglas desde normativa PDF
curl -X POST http://localhost:8000/rules/extract \
  -F "pdf=@/ruta/normativa_iso.pdf" \
  -F "profile_name=iso_11228"
```

### Uso programático (Python)

```python
from schemas import VideoInput, ProcessingMode
from pose_pipeline import PosePipeline
from ergo_engine import ErgoEngine
from reports import ReportGenerator

# Pipeline completo sin API
video_input = VideoInput(
    path="/ruta/al/video.mp4",
    person_height_cm=175.0,
    ergo_methods=["REBA", "RULA", "OWAS"],
    fps_sample_rate=1,          # analizar todos los frames
    processing_mode=ProcessingMode.CPU_ONLY,
)

skeleton_seq = PosePipeline(device="cpu").process(video_input)
frame_scores = ErgoEngine(methods=["REBA", "RULA", "OWAS"]).analyze(skeleton_seq)
ReportGenerator().generate(build_report(frame_scores), "output/reporte.pdf")

# Con GPU Enhanced (requiere GVHMR instalado)
from advanced_pipeline import AdvancedPosePipeline
skeleton_seq = AdvancedPosePipeline(device="cuda").process(video_input)
```

### Extraer reglas LLM desde PDF

```python
from llm_rules import RuleExtractor

extractor = RuleExtractor()
rules = extractor.extract("normativa_iso_11228.pdf", profile_name="iso_11228")
# Genera: iso_11228_rules.json (usar en ergo_methods=["LLM"] con rules_profile_path)
```

---

## Tests

```bash
# Todos los tests (243 tests, ~8s)
pytest --tb=short -m "not gpu and not integration" -q

# Por módulo
pytest schemas/ -v
pytest ergo_engine/ -v
pytest pose_pipeline/ -v
pytest llm_rules/ -v
pytest reports/ -v
pytest advanced_pipeline/ -v
pytest api/ -v

# Test específico
pytest ergo_engine/tests/test_reba.py::test_reba_standing_upright -v

# Con tests de integración (requiere Ollama corriendo)
pytest llm_rules/ -v -m integration

# Con tests de GPU (requiere CUDA)
pytest advanced_pipeline/ -v -m gpu
```

---

## Variables de Entorno

```bash
OLLAMA_HOST=http://localhost:11434   # URL del servidor Ollama
MODEL_DIR=./models                   # Directorio de modelos ONNX
LOG_LEVEL=INFO                       # DEBUG | INFO | WARNING | ERROR
```

Crear `.env` en la raíz del proyecto para persistirlas.

---

## Arquitectura y Dependencias entre Módulos

```
schemas           ← TODOS dependen de este. Única fuente de verdad de tipos.
pose_pipeline     ← depende solo de: schemas
ergo_engine       ← depende solo de: schemas
llm_rules         ← depende solo de: schemas
reports           ← depende solo de: schemas
advanced_pipeline ← depende de: schemas + pose_pipeline
api               ← depende de: todos los anteriores (único orquestador)
```

**Regla de oro:** Ningún módulo importa de otro módulo excepto de `schemas` y sus dependencias explícitas.

## Flujo de Datos Principal

```
VideoInput → PosePipeline → SkeletonSequence
                                    ↓
                            ErgoEngine.analyze()
                                    ↓
                        list[FrameErgonomicScore]
                                    ↓
                      ReportGenerator.generate() → PDF
```

`AnalysisOrchestrator` (`api/services/orchestrator.py`) es el único lugar que une estas tres fases.

## Gestión de GPU (CRÍTICO)

El sistema tiene dos fases que usan GPU — **nunca simultáneas:**

1. **Fase LLM** (`llm_rules`): Gemma 3 4B via Ollama (~2.5 GB VRAM). Al terminar, libera GPU con `keep_alive: 0` vía `GPUManager.release_llm_vram()`.
2. **Fase visión** (`pose_pipeline` / `advanced_pipeline`): usa la GPU liberada.

`GPUManager.release_llm_vram()` se llama en bloque `finally` — siempre, incluso si hay error.

## Motores de Estimación de Postura (advanced_pipeline)

El `PipelineRouter` selecciona automáticamente el motor:

| Motor | Cuándo usar | Estado |
|---|---|---|
| **GVHMR** | default GPU, cámara móvil, >5 min | Wrapper implementado — requiere `pip install -e GVHMR/` |
| **WHAM** | análisis de marcha, <2 min | Stub — DPVO no compila en Windows |
| **TRAM** | si GVHMR falla, GPU≥4GB | Stub — requiere `pip install -e tram/` |
| **HumanMM** | video multi-plano (shot cuts) | Stub — requiere `pip install -e HumanMM-code/` |
| **MotionBERT Lite** | fallback CPU siempre disponible | Implementado, descarga automática |

Todos los motores implementan `is_available() -> bool`. `AdvancedPosePipeline` cae en `PosePipeline` (CPU) si ningún motor GPU está disponible.

## Convenciones de Código

- Python 3.11+, type hints obligatorios en funciones públicas
- Pydantic v2 para todos los modelos de datos
- `loguru` para logging (nunca `print`)
- `pytest` con fixtures en `conftest.py` de cada módulo
- Marcadores de test: `@pytest.mark.gpu` y `@pytest.mark.integration`
- Tests unitarios usan mocks ONNX (`patch("onnxruntime.InferenceSession", ...)`) — los modelos son lazy-loaded

## Notas de Implementación Clave

- Los modelos ONNX se descargan en `MODEL_DIR` al primer uso (lazy loading vía `model_downloader.py`). Sin modelos, `PosePipeline` usa modo sintético (posturas proporcionales generadas algorítmicamente).
- `SkeletonSequence` con `coordinate_system="camera"` en CPU_ONLY, `"world"` en GPU_ENHANCED.
- `schemas/` es inmutable una vez publicado: renombrar campos es un breaking change para todos los módulos.
- WeasyPrint (reports) requiere GTK3 runtime en Windows. El generador auto-detecta el backend: WeasyPrint → reportlab → HTML-only.
- WHAM/DPVO no compilan en Windows; usar `contact_refinement.py` para análisis de contacto pie-suelo.
- La caché de reglas LLM está en `.ergo_cache/` (SHA-256 del PDF). Borrar para forzar re-extracción.

## Hardware Objetivo

- CPU: Intel i7 (8+ cores), RAM: 16 GB
- GPU: 4 GB VRAM NVIDIA (base), 6-8 GB para motores avanzados (TRAM/GVHMR)
- OS: Windows 10/11
