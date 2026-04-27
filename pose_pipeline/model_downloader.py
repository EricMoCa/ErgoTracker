"""
pose_pipeline/model_downloader.py
Descarga lazy de modelos ONNX al directorio MODEL_DIR.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Registro de modelos conocidos
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        "filename": "yolov8n.onnx",
        "sha256": None,  # verificación opcional
    },
    "rtmpose_m": {
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
        ),
        "filename": "rtmpose-m.onnx",
        "sha256": None,
    },
    "motionbert_lite": {
        "url": "https://github.com/Walter0807/MotionBERT/releases/download/v0.1/motionbert_lite.onnx",
        "filename": "motionbert_lite.onnx",
        "sha256": None,
    },
}


def _get_model_dir() -> Path:
    """Retorna el directorio de modelos desde la variable de entorno MODEL_DIR."""
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _verify_sha256(path: Path, expected: str) -> bool:
    """Verifica la integridad SHA-256 de un archivo."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def ensure_model(name: str) -> Path:
    """
    Descarga el modelo si no existe localmente. Retorna el path local.

    Args:
        name: clave del modelo en el diccionario MODELS.

    Returns:
        Path al archivo del modelo descargado.

    Raises:
        ValueError: si el modelo no está registrado.
        RuntimeError: si la descarga falla.
    """
    if name not in MODELS:
        raise ValueError(
            f"Modelo '{name}' no registrado. Opciones: {list(MODELS.keys())}"
        )

    model_info = MODELS[name]
    model_dir = _get_model_dir()
    model_path = model_dir / model_info["filename"]

    if model_path.exists():
        logger.debug(f"Modelo '{name}' ya existe en {model_path}")
        # Verificar integridad si hay sha256
        if model_info.get("sha256"):
            if _verify_sha256(model_path, model_info["sha256"]):
                logger.debug(f"SHA-256 verificado para '{name}'")
            else:
                logger.warning(
                    f"SHA-256 mismatch para '{name}', re-descargando..."
                )
                model_path.unlink()
                return _download_model(name, model_info, model_path)
        return model_path

    return _download_model(name, model_info, model_path)


def _download_model(name: str, model_info: dict, model_path: Path) -> Path:
    """Realiza la descarga efectiva del modelo con barra de progreso."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        raise RuntimeError(
            f"Dependencias de descarga no disponibles: {e}. "
            "Instalar con: pip install requests tqdm"
        ) from e

    url = model_info["url"]
    logger.info(f"Descargando modelo '{name}' desde {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Error al descargar el modelo '{name}': {e}") from e

    total_size = int(response.headers.get("content-length", 0))
    with open(model_path, "wb") as f, tqdm(
        desc=name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    logger.success(f"Modelo '{name}' descargado en {model_path}")

    # Verificar sha256 si está disponible
    if model_info.get("sha256"):
        if not _verify_sha256(model_path, model_info["sha256"]):
            model_path.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch para el modelo '{name}' descargado. "
                "Archivo eliminado."
            )

    return model_path
