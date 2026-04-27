"""
api/routes/setup.py
Endpoints para verificar el estado del entorno y descargar modelos ONNX.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from ..config import settings
from ..storage.job_store import JobStatus, JobStore

router = APIRouter(prefix="/setup", tags=["setup"])
_job_store = JobStore()

# progreso por job_id: {"downloaded": int, "total": int, "pct": float}
_progress: dict[str, dict] = {}

_PACKAGES = [
    ("cv2",          "opencv-python"),
    ("onnxruntime",  "onnxruntime"),
    ("numpy",        "numpy"),
    ("torch",        "torch (opcional)"),
]

_ONNX_MODELS = {
    "yolov8n":        "yolov8n.onnx",
    "rtmpose_m":      "rtmpose-m.onnx",
    "motionbert_lite": "motionbert_lite.onnx",
}


@router.get("/status")
async def get_setup_status():
    """
    Devuelve el estado de todos los prerrequisitos:
      - paquetes Python (opencv, onnxruntime, numpy, torch)
      - modelos ONNX (existencia + tamaño)
      - servidor Ollama + modelo gemma3:4b
    """
    model_dir = settings.model_dir

    packages = {}
    for module_name, pkg_label in _PACKAGES:
        try:
            importlib.import_module(module_name)
            packages[module_name] = {"available": True, "label": pkg_label}
        except ImportError:
            packages[module_name] = {"available": False, "label": pkg_label}

    models = {}
    for key, filename in _ONNX_MODELS.items():
        path = model_dir / filename
        exists = path.exists()
        models[key] = {
            "available": exists,
            "filename": filename,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1) if exists else 0,
        }

    ollama_ok = False
    gemma_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                listed = resp.json().get("models", [])
                gemma_ok = any("gemma3" in m.get("name", "") for m in listed)
    except Exception:
        pass

    return {
        "packages": packages,
        "models": models,
        "ollama": {"available": ollama_ok, "gemma3_4b": gemma_ok},
        "model_dir": str(model_dir),
    }


@router.post("/download/{model_name}", status_code=202)
async def start_download(model_name: str, background_tasks: BackgroundTasks):
    """
    Inicia la descarga de un modelo ONNX en background.
    Retorna job_id para hacer polling de progreso en GET /setup/download/{job_id}.
    """
    from pose_pipeline.model_downloader import MODELS

    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no registrado. Disponibles: {list(MODELS.keys())}",
        )

    job = _job_store.create("model_download", {"model_name": model_name})
    _progress[job.id] = {"downloaded": 0, "total": 0, "pct": 0}
    background_tasks.add_task(_run_download, job.id, model_name)
    return {"job_id": job.id, "status": "pending", "model": model_name}


@router.get("/download/{job_id}")
async def get_download_status(job_id: str):
    """Polling del estado y progreso de una descarga en curso."""
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": _progress.get(job_id, {}),
        "error": job.error,
    }


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_download(job_id: str, model_name: str) -> None:
    from pose_pipeline.model_downloader import ensure_model

    _job_store.update_status(job_id, JobStatus.RUNNING)
    logger.info(f"[setup] Iniciando descarga de '{model_name}' (job {job_id})")

    def _on_progress(downloaded: int, total: int) -> None:
        pct = round(downloaded / total * 100, 1) if total > 0 else 0
        _progress[job_id] = {"downloaded": downloaded, "total": total, "pct": pct}

    try:
        ensure_model(model_name, progress_callback=_on_progress)
        _progress[job_id]["pct"] = 100
        _job_store.update_status(job_id, JobStatus.COMPLETED, result={"model": model_name})
        logger.success(f"[setup] Descarga completada: '{model_name}'")
    except Exception as exc:
        logger.error(f"[setup] Descarga fallida para '{model_name}': {exc}")
        _job_store.update_status(job_id, JobStatus.FAILED, error=str(exc))
