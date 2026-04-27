# Task D — api/

## Objetivo
Implementar el backend FastAPI que orquesta todos los módulos del sistema. Es el único punto de entrada externo y el único módulo que importa de todos los demás.

## Dependencias previas
**REQUIERE que Tasks A (pose_pipeline), B (ergo_engine), C (llm_rules) y E (reports) estén completos.** Este módulo es el último en implementarse salvo `advanced_pipeline`.

## Archivos a crear

```
api/
├── __init__.py
├── main.py                   # FastAPI app, lifespan, middleware
├── config.py                 # Settings desde variables de entorno
├── routes/
│   ├── __init__.py
│   ├── analysis.py           # POST /analyze
│   ├── rules.py              # POST /rules/extract, GET /rules, DELETE /rules/{id}
│   └── reports.py            # GET /reports/{id}, GET /reports/{id}/download
├── services/
│   ├── __init__.py
│   └── orchestrator.py       # Lógica de orquestación: pose → ergo → report
├── storage/
│   ├── __init__.py
│   └── job_store.py          # In-memory store de jobs (sin base de datos)
├── conftest.py               # Fixtures con TestClient y mocks
└── tests/
    ├── test_analysis_route.py
    ├── test_rules_route.py
    └── test_reports_route.py
```

## Implementación por archivo

### config.py
```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    model_dir: Path = Path("./models")
    log_level: str = "INFO"
    max_video_size_mb: int = 500
    default_person_height_cm: float = 170.0
    reports_output_dir: Path = Path("./output/reports")

    class Config:
        env_file = ".env"

settings = Settings()
```

### storage/job_store.py
```python
from enum import Enum
from datetime import datetime
from typing import Any
import uuid

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"

class Job:
    def __init__(self, job_type: str, params: dict):
        self.id = str(uuid.uuid4())
        self.job_type = job_type
        self.params = params
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.result: Any = None
        self.error: str | None = None

class JobStore:
    """
    Store en memoria para jobs de análisis.
    Permite polling del estado del job desde la UI.
    Sin base de datos — los jobs se pierden al reiniciar el servidor.
    """
    _jobs: dict[str, Job] = {}

    def create(self, job_type: str, params: dict) -> Job:
        job = Job(job_type, params)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, result=None, error=None):
        job = self._jobs.get(job_id)
        if job:
            job.status = status
            if result: job.result = result
            if error: job.error = error

    def list_all(self) -> list[Job]:
        return list(self._jobs.values())
```

### services/orchestrator.py
```python
from schemas import VideoInput, AnalysisReport
from pose_pipeline import PosePipeline
from ergo_engine import ErgoEngine
from reports import ReportGenerator
from .config import settings
from loguru import logger

class AnalysisOrchestrator:
    """
    Orquesta el pipeline completo de análisis:
    1. PosePipeline → SkeletonSequence
    2. ErgoEngine   → list[FrameErgonomicScore]
    3. ReportGenerator → PDF

    GESTIÓN DE GPU:
    - Si rules_json_path es None y se solicita extracción LLM, ese paso debe haberse
      completado ANTES de llamar al orquestador (la GPU ya debe estar libre).
    - El orquestador NO llama a RuleExtractor directamente.
    """

    def __init__(self, device: str = "cpu"):
        self.pose_pipeline = PosePipeline(device=device)
        self.report_generator = ReportGenerator()

    def run(self, video_input: VideoInput) -> AnalysisReport:
        """
        Ejecuta el pipeline completo y retorna el AnalysisReport.
        Crea y guarda el PDF automáticamente en settings.reports_output_dir.
        """
        logger.info(f"Iniciando análisis: {video_input.path}")

        # 1. Estimación de postura
        logger.info("Fase 1: Estimando postura 3D...")
        skeleton_seq = self.pose_pipeline.process(video_input)

        # 2. Análisis ergonómico
        logger.info("Fase 2: Analizando ergonomía...")
        ergo_engine = ErgoEngine(
            methods=video_input.ergo_methods,
            rules_json_path=video_input.rules_profile_path
        )
        frame_scores = ergo_engine.analyze(skeleton_seq)

        # 3. Construir AnalysisReport
        report = self._build_report(video_input, skeleton_seq, frame_scores)

        # 4. Generar PDF
        pdf_path = settings.reports_output_dir / f"{report.id}.pdf"
        self.report_generator.generate(report, str(pdf_path))
        logger.success(f"Análisis completado. Reporte: {pdf_path}")

        return report

    def _build_report(self, video_input, skeleton_seq, frame_scores) -> AnalysisReport:
        """Construye el AnalysisReport desde los resultados de cada módulo."""
        from schemas import AnalysisReport, ReportSummary, RiskLevel
        from datetime import datetime
        import uuid

        # Calcular summary
        high_risk_frames = sum(
            1 for f in frame_scores
            if f.overall_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        )
        max_reba = max((f.reba.total for f in frame_scores if f.reba), default=None)

        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            video_path=video_input.path,
            duration_s=len(skeleton_seq.frames) / skeleton_seq.fps,
            total_frames=skeleton_seq.total_frames,
            analyzed_frames=len(frame_scores),
            person_height_cm=video_input.person_height_cm,
            methods_used=video_input.ergo_methods,
            frame_scores=frame_scores,
            summary=ReportSummary(
                max_reba_score=max_reba,
                pct_frames_high_risk=high_risk_frames / max(len(frame_scores), 1),
            )
        )
```

### routes/analysis.py
```python
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from schemas import VideoInput, ProcessingMode
from ..services.orchestrator import AnalysisOrchestrator
from ..storage.job_store import JobStore, JobStatus
import shutil, tempfile
from pathlib import Path

router = APIRouter(prefix="/analyze", tags=["analysis"])
job_store = JobStore()
orchestrator = AnalysisOrchestrator()

@router.post("/", status_code=202)
async def start_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    person_height_cm: float = Form(default=170.0),
    ergo_methods: str = Form(default="REBA"),       # CSV: "REBA,RULA"
    rules_profile_path: str = Form(default=None),
    processing_mode: str = Form(default="cpu_only"),
):
    """
    Inicia análisis ergonómico en background.
    Retorna job_id para hacer polling del estado.
    """
    # Guardar video temporalmente
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    video_input = VideoInput(
        path=str(video_path),
        person_height_cm=person_height_cm,
        ergo_methods=ergo_methods.split(","),
        rules_profile_path=rules_profile_path,
        processing_mode=ProcessingMode(processing_mode),
    )

    job = job_store.create("analysis", {"video_path": str(video_path)})
    background_tasks.add_task(_run_analysis, job.id, video_input)
    return {"job_id": job.id, "status": job.status}

@router.get("/{job_id}")
async def get_analysis_status(job_id: str):
    """Polling del estado del job de análisis."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result.model_dump() if job.result else None,
        "error": job.error
    }

async def _run_analysis(job_id: str, video_input: VideoInput):
    job_store.update_status(job_id, JobStatus.RUNNING)
    try:
        report = orchestrator.run(video_input)
        job_store.update_status(job_id, JobStatus.COMPLETED, result=report)
    except Exception as e:
        job_store.update_status(job_id, JobStatus.FAILED, error=str(e))
```

### routes/rules.py
```python
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from llm_rules import RuleExtractor
from ..storage.job_store import JobStore, JobStatus
import shutil, tempfile
from pathlib import Path

router = APIRouter(prefix="/rules", tags=["rules"])
job_store = JobStore()
extractor = RuleExtractor()

@router.post("/extract", status_code=202)
async def extract_rules(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    profile_name: str = Form(default="custom"),
):
    """
    Extrae reglas ergonómicas de un PDF usando Gemma 3 4B.
    IMPORTANTE: Este endpoint usa la GPU. No llamar mientras hay un análisis de video en curso.
    Retorna job_id para hacer polling. Al completar, la GPU queda libre.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / pdf.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    job = job_store.create("rule_extraction", {"pdf": pdf.filename})
    background_tasks.add_task(_run_extraction, job.id, str(pdf_path), profile_name)
    return {"job_id": job.id, "status": "pending"}

@router.get("/{job_id}")
async def get_extraction_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return {"job_id": job.id, "status": job.status, "result": job.result, "error": job.error}

async def _run_extraction(job_id: str, pdf_path: str, profile_name: str):
    job_store.update_status(job_id, JobStatus.RUNNING)
    try:
        rules = extractor.extract(pdf_path, profile_name)
        job_store.update_status(job_id, JobStatus.COMPLETED,
            result={"rules_count": len(rules), "profile": f"{profile_name}_rules.json"})
    except Exception as e:
        job_store.update_status(job_id, JobStatus.FAILED, error=str(e))
```

### routes/reports.py
```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..config import settings

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """Descarga el PDF de un reporte ya generado."""
    pdf_path = settings.reports_output_dir / f"{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Reporte no encontrado")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        filename=f"reporte_ergo_{report_id}.pdf")
```

### main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from .config import settings
from .routes import analysis, rules, reports

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: crear directorios necesarios
    settings.reports_output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ErgoTracker API iniciada")
    yield
    logger.info("ErgoTracker API detenida")

app = FastAPI(
    title="ErgoTracker API",
    description="Análisis ergonómico post-proceso desde video monocular",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],   # En producción: restringir al dominio de la UI
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(rules.router)
app.include_router(reports.router)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

## Interfaz Pública
```python
# api/__init__.py
from .main import app
__all__ = ["app"]
```

## Tests

### conftest.py
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_orchestrator(monkeypatch):
    """Mock del orquestador para tests de rutas sin ejecutar el pipeline real."""
    ...
```

### test_analysis_route.py
- `test_health_endpoint`: GET /health → 200
- `test_start_analysis_returns_job_id`: POST /analyze con video sintético → 202 + job_id
- `test_poll_job_status`: GET /analyze/{job_id} retorna status
- `test_analysis_completes`: con mock del orquestador, job llega a COMPLETED

## Dependencias

```
# requirements/api.txt
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9    # Para UploadFile
pydantic-settings>=2.0.0
loguru>=0.7.0
httpx>=0.26.0              # Para TestClient
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## Comandos

```bash
cd api
pip install -r ../requirements/api.txt

# Correr servidor:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Tests:
pytest tests/ -v

# Ver docs interactivos:
# Abrir http://localhost:8000/docs en el navegador
```

## NO HACER

- No implementar lógica de análisis aquí (solo orquestar)
- No acceder a modelos ONNX directamente
- No conectar a Ollama directamente (usar llm_rules/)
- No modificar archivos en `schemas/`
- No añadir base de datos (usar JobStore en memoria para v1)
