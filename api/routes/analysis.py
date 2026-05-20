import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from schemas import VideoInput, ProcessingMode
from ..storage.job_store import JobStore, JobStatus
from ..services.orchestrator import AnalysisOrchestrator
from ..config import settings

router = APIRouter(prefix="/analyze", tags=["analysis"])

_job_store = JobStore()
_orchestrator = AnalysisOrchestrator()


@router.post("/", status_code=202)
async def start_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    person_height_cm: float = Form(default=170.0),
    ergo_methods: str = Form(default="REBA"),
    rules_profile_path: str = Form(default=None),
    processing_mode: str = Form(default="cpu_only"),
    preferred_engine: str = Form(default="auto"),
    requires_gait_analysis: bool = Form(default=False),
    has_multiple_shots: bool = Form(default=False),
    camera_motion_high: bool = Form(default=False),
):
    """
    Starts an ergonomic analysis job in the background.
    Returns a job_id for status polling via GET /analyze/{job_id}.
    ergo_methods: comma-separated list — e.g. "REBA,RULA"
    preferred_engine: auto | gvhmr | wham | tram | humanmm (only with gpu_enhanced)
    """
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / (video.filename or "upload.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    video_input = VideoInput(
        path=str(video_path),
        person_height_cm=person_height_cm,
        ergo_methods=ergo_methods.split(","),
        rules_profile_path=rules_profile_path,
        processing_mode=ProcessingMode(processing_mode),
        preferred_engine=preferred_engine,
        requires_gait_analysis=requires_gait_analysis,
        has_multiple_shots=has_multiple_shots,
        camera_motion_high=camera_motion_high,
    )

    job = _job_store.create("analysis", {"video_path": str(video_path)})
    background_tasks.add_task(_run_analysis, job.id, video_input)
    return {"job_id": job.id, "status": job.status}


@router.get("/{job_id}")
async def get_analysis_status(job_id: str):
    """Returns current status of an analysis job."""
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "result": job.result.model_dump() if job.result else None,
        "error": job.error,
    }


async def _run_analysis(job_id: str, video_input: VideoInput) -> None:
    _job_store.update_status(job_id, JobStatus.RUNNING)
    _job_store.update_progress(job_id, 0, "Iniciando...")

    def on_progress(pct: float, stage: str) -> None:
        _job_store.update_progress(job_id, pct, stage)

    try:
        report = _orchestrator.run(
            video_input,
            output_dir=str(settings.reports_output_dir),
            progress_callback=on_progress,
        )
        _job_store.update_status(job_id, JobStatus.COMPLETED, result=report)
    except Exception as exc:
        _job_store.update_status(job_id, JobStatus.FAILED, error=str(exc))
