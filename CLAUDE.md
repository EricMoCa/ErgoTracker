# ErgoTracker — Proyecto Principal

## Descripción
Plataforma modular de análisis ergonómico post-proceso: a partir de un video monocular extrae
posturas 3D en coordenadas de mundo y evalúa riesgos muscoesqueléticos (REBA, RULA, OWAS,
reglas LLM). Corre completamente offline en laptops empresariales (i7, 4 GB VRAM).

## Estructura del Repositorio

```
ergotracker/
├── schemas/            # Task 0 — Contratos de datos compartidos (FOUNDATION)
├── pose_pipeline/      # Task A — Estimación de postura 2D/3D (RTMPose + MotionBERT)
├── ergo_engine/        # Task B — Motor de análisis ergonómico (REBA/RULA/OWAS)
├── llm_rules/          # Task C — Extracción de reglas desde PDF con Gemma/Ollama
├── api/                # Task D — Backend FastAPI (orquestador)
├── reports/            # Task E — Generador de reportes PDF
├── advanced_pipeline/  # Task F — Pipeline avanzado con motores intercambiables
│   ├── gvhmr_wrapper.py        # Motor primario world-grounded (SIGGRAPH Asia 2024)
│   ├── wham_wrapper.py         # Motor secundario — análisis de contacto pie-suelo
│   ├── tram_wrapper.py         # Motor alternativo — SLAM + VIMO (ECCV 2024)
│   ├── contact_refinement.py   # Refinamiento post-proceso: contacto pie-suelo
│   ├── stride_refinement.py    # Refinamiento por oclusión (WACV 2025)
│   ├── humanmm_wrapper.py      # Multi-shot video (CVPR 2025)
│   ├── pipeline_router.py      # Selector automático de motor según video + hardware
│   └── pipeline.py             # AdvancedPosePipeline — orquestador
└── requirements/       # Dependencias por módulo
```

## Dependencias entre Módulos (NO violar)

```
schemas           ← TODOS dependen de este. Es la única dependencia compartida.
pose_pipeline     ← depende de: schemas
ergo_engine       ← depende de: schemas
llm_rules         ← depende de: schemas
reports           ← depende de: schemas
advanced_pipeline ← depende de: schemas + pose_pipeline (interfaz pública)
api               ← depende de: todos los anteriores (es el orquestador final)
```

**REGLA DE ORO: Ningún módulo importa de otro módulo excepto de `schemas`
y sus dependencias explícitas declaradas arriba.**

## Arquitectura de Motores Intercambiables

`advanced_pipeline` expone motores world-grounded como plugins con interfaz común.
El `PipelineRouter` selecciona automáticamente el motor según el video y el hardware.

### Motores world-grounded disponibles (en orden de prioridad)

| Motor | Venue | Ventaja clave | Cuándo usar |
|---|---|---|---|
| **GVHMR** | SIGGRAPH Asia 2024 | Sin drift, 7× más rápido, mejor en benchmarks world-grounded | Default para cámara móvil |
| **WHAM** | CVPR 2024 | Contacto pie-suelo, eliminación de foot sliding | Análisis de marcha/carga |
| **TRAM** | ECCV 2024 | SLAM + VIMO (ViT-H), 60% menos error en trayectoria | Alternativa si GVHMR falla |
| **MotionBERT Lite** | ICCV 2023 | CPU-only, sin GPU, siempre disponible | Baseline y fallback |

### Estrategia de selección del PipelineRouter

```
video.duration > 5 min       → GVHMR  (sin drift en secuencias largas)
video.has_gait_analysis      → GVHMR + contact_refinement (foot contact de WHAM)
video.has_multiple_shots     → HumanMM + GVHMR
video.camera_moving + GPU≥6G → GVHMR (o TRAM como alternativa)
video.occlusion_high         → cualquier motor + STRIDE refinement
fallback                     → MotionBERT Lite (CPU)
```

### Formato interno: ErgoPose

Todos los motores producen `SkeletonSequence` con `coordinate_system="world"`.
Los campos de calidad en `Skeleton3D` permiten que el motor ergonómico sea
agnóstico al motor de visión usado (GVHMR, WHAM, TRAM, MotionBERT).

## Pipeline de GPU (CRÍTICO)

El sistema tiene dos fases secuenciales que usan GPU:

1. **Fase A — LLM** (`llm_rules`): Gemma 3 4B via Ollama (~2.5 GB VRAM).
   Al terminar, libera la GPU explícitamente con `keep_alive: 0`.
2. **Fase B — Visión** (`pose_pipeline` / `advanced_pipeline`): Usa la GPU
   liberada para GVHMR / WHAM / TRAM según el perfil seleccionado.

**Nunca cargar LLM y modelos de visión en GPU simultáneamente.**

## Convenciones de Código

- Python 3.11+
- Type hints obligatorios en todas las funciones públicas
- Pydantic v2 para todos los modelos de datos
- pytest para tests, con fixtures en `conftest.py`
- Logging con `loguru`, no `print`
- Docstrings en español o inglés (consistente dentro de cada módulo)
- Black + isort para formateo
- Cada motor tiene `is_available() -> bool` y retorna `None` si no está instalado

## Comandos Globales

```bash
# Instalar dependencias base
pip install -r requirements/base.txt

# Correr todos los tests
pytest --tb=short

# Correr tests por módulo (para desarrollo paralelo)
pytest schemas/       --tb=short
pytest pose_pipeline/ --tb=short
pytest ergo_engine/   --tb=short
pytest llm_rules/     --tb=short
pytest reports/       --tb=short
pytest advanced_pipeline/ --tb=short

# Verificar tipos
mypy . --ignore-missing-imports
```

## Variables de Entorno

```bash
OLLAMA_HOST=http://localhost:11434   # URL del servidor Ollama
MODEL_DIR=./models                   # Directorio de modelos ONNX descargados
LOG_LEVEL=INFO
```

## Hardware Objetivo

- CPU: Intel i7 (8+ cores)
- GPU: 4 GB VRAM NVIDIA (base), 6-8 GB para motores avanzados
- RAM: 16 GB
- OS: Windows 10/11

## Orden de Implementación para Agentes Paralelos

```
Fase 1 (secuencial):   schemas/
Fase 2 (paralelo):     pose_pipeline/ | ergo_engine/ | llm_rules/ | reports/
Fase 3 (secuencial):   advanced_pipeline/   ← espera a que pose_pipeline esté listo
Fase 4 (secuencial):   api/                 ← espera a que TODOS estén listos
```

## Notas de Arquitectura

- Los modelos ONNX/PyTorch se descargan en `MODEL_DIR` al primer uso (lazy loading)
- Todos los módulos exponen interfaz pública mínima en su `__init__.py`
- Los schemas son inmutables — cambios deben coordinar con todos los módulos
- Cada motor avanzado (GVHMR, WHAM, TRAM) es un stub que retorna `None` si no
  está instalado, permitiendo que `api/` funcione sin ellos
