"""
pose_pipeline/model_downloader.py
Descarga lazy de modelos ONNX al directorio MODEL_DIR.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Registro de modelos conocidos
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        "filename": "yolov8n.onnx",
        "sha256": None,
    },
    "rtmpose_m": {
        # La URL oficial de OpenMMLab distribuye un ZIP con el ONNX dentro
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
        ),
        "filename": "rtmpose-m.onnx",
        # Nombre del archivo ONNX dentro del ZIP
        "zip_inner": "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx",
        "sha256": None,
    },
    "motionbert_lite": {
        # Fuente primaria: Hugging Face (más estable que GitHub releases)
        "url": "https://huggingface.co/walterzhu/MotionBERT/resolve/main/MB_lite.onnx",
        # Alternativa si la primaria falla:
        "url_fallback": "https://github.com/Walter0807/MotionBERT/releases/download/v0.1/motionbert_lite.onnx",
        "filename": "motionbert_lite.onnx",
        "sha256": None,
    },
}


def _get_model_dir() -> Path:
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _verify_sha256(path: Path, expected: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def ensure_model(name: str, progress_callback=None) -> Path:
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
        if model_info.get("sha256"):
            if _verify_sha256(model_path, model_info["sha256"]):
                logger.debug(f"SHA-256 verificado para '{name}'")
            else:
                logger.warning(f"SHA-256 mismatch para '{name}', re-descargando...")
                model_path.unlink()
                return _download_model(name, model_info, model_path, progress_callback=progress_callback)
        return model_path

    return _download_model(name, model_info, model_path, progress_callback=progress_callback)


def _download_model(
    name: str,
    model_info: dict,
    model_path: Path,
    progress_callback=None,
) -> Path:
    """Realiza la descarga del modelo con soporte para ZIPs y fallback URLs.

    progress_callback: callable(downloaded_bytes: int, total_bytes: int) | None
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        raise RuntimeError(
            f"Dependencias de descarga no disponibles: {e}. "
            "Instalar con: pip install requests tqdm"
        ) from e

    urls: list[str] = [model_info["url"]]
    if model_info.get("url_fallback"):
        urls.append(model_info["url_fallback"])

    last_error: Exception | None = None
    for url in urls:
        logger.info(f"Descargando modelo '{name}' desde {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            is_zip = (
                url.lower().endswith(".zip")
                or "application/zip" in response.headers.get("content-type", "")
                or "application/x-zip" in response.headers.get("content-type", "")
            )

            if is_zip:
                model_path = _download_and_extract_zip(
                    name, response, total_size, model_path, model_info,
                    progress_callback=progress_callback,
                )
            else:
                downloaded = 0
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
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded, total_size)

            logger.success(f"Modelo '{name}' descargado en {model_path}")

            if model_info.get("sha256"):
                if not _verify_sha256(model_path, model_info["sha256"]):
                    model_path.unlink()
                    raise RuntimeError(
                        f"SHA-256 mismatch para el modelo '{name}' descargado."
                    )

            return model_path

        except Exception as e:
            logger.warning(f"Descarga fallida desde {url}: {e}")
            last_error = e
            if model_path.exists():
                model_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"No se pudo descargar el modelo '{name}' desde ninguna URL. "
        f"Último error: {last_error}"
    )


def _download_and_extract_zip(
    name: str,
    response,
    total_size: int,
    model_path: Path,
    model_info: dict,
    progress_callback=None,
) -> Path:
    """Descarga un ZIP y extrae el ONNX especificado en zip_inner."""
    import tempfile
    from tqdm import tqdm

    zip_inner: str | None = model_info.get("zip_inner")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip_path = Path(tmp_zip.name)
        downloaded = 0
        with tqdm(
            desc=f"{name} (zip)",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_zip.write(chunk)
                    bar.update(len(chunk))
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    try:
        with zipfile.ZipFile(tmp_zip_path, "r") as zf:
            members = zf.namelist()
            logger.debug(f"ZIP contiene: {members}")

            # Si se conoce el nombre exacto dentro del ZIP, usarlo
            if zip_inner and zip_inner in members:
                target = zip_inner
            else:
                # Buscar cualquier .onnx dentro del ZIP
                onnx_members = [m for m in members if m.lower().endswith(".onnx")]
                if not onnx_members:
                    raise RuntimeError(
                        f"No se encontró ningún archivo .onnx en el ZIP de '{name}'. "
                        f"Contenido: {members}"
                    )
                target = onnx_members[0]
                if len(onnx_members) > 1:
                    logger.warning(
                        f"Múltiples ONNX en ZIP, usando el primero: {target}"
                    )

            # Extraer directamente al path de destino
            with zf.open(target) as src, open(model_path, "wb") as dst:
                dst.write(src.read())

        logger.info(f"Extraído '{target}' → {model_path}")
    finally:
        tmp_zip_path.unlink(missing_ok=True)

    return model_path
