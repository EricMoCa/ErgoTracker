import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from llm_rules import RuleExtractor
from ..storage.job_store import JobStore, JobStatus

router = APIRouter(prefix="/rules", tags=["rules"])

_job_store = JobStore()
_extractor = RuleExtractor()


@router.post("/extract", status_code=202)
async def extract_rules(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    profile_name: str = Form(default="custom"),
):
    """
    Extracts ergonomic rules from a PDF using Gemma 3 4B (via Ollama).
    IMPORTANT: Uses the GPU. Do not call while a video analysis job is running.
    Returns a job_id for status polling. On completion, GPU VRAM is released.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / (pdf.filename or "rules.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    job = _job_store.create("rule_extraction", {"pdf": pdf.filename})
    background_tasks.add_task(_run_extraction, job.id, str(pdf_path), profile_name)
    return {"job_id": job.id, "status": "pending"}


@router.get("/{job_id}")
async def get_extraction_status(job_id: str):
    """Returns status of a rule extraction job."""
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }


async def _run_extraction(job_id: str, pdf_path: str, profile_name: str) -> None:
    _job_store.update_status(job_id, JobStatus.RUNNING)
    try:
        rules = _extractor.extract(pdf_path, profile_name)
        _job_store.update_status(
            job_id,
            JobStatus.COMPLETED,
            result={"rules_count": len(rules), "profile": f"{profile_name}_rules.json"},
        )
    except Exception as exc:
        _job_store.update_status(job_id, JobStatus.FAILED, error=str(exc))
