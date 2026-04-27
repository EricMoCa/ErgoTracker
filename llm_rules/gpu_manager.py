"""
llm_rules/gpu_manager.py
Gestión de liberación de VRAM después de inferencia LLM.
"""
import os

import requests
from loguru import logger

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class GPUManager:
    """
    Gestiona la liberación de VRAM después de usar el LLM.
    Patrón: cargar Gemma → extraer reglas → liberar → pipeline de visión usa GPU libre.
    """

    @staticmethod
    def release_llm_vram(model_name: str = "gemma3:4b") -> None:
        """
        Fuerza a Ollama a descargar el modelo de la GPU enviando keep_alive=0.
        Llama siempre al finalizar cualquier operación LLM.
        """
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"GPU VRAM liberada: modelo {model_name} descargado de Ollama")
        except Exception as e:
            logger.warning(f"No se pudo liberar VRAM via Ollama: {e}")

        # Purgar caché CUDA residual si PyTorch está disponible
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() ejecutado")
        except ImportError:
            pass  # PyTorch no requerido para este módulo

    @staticmethod
    def is_ollama_running() -> bool:
        """Verifica si el servidor Ollama está activo."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """Verifica si el modelo está descargado en Ollama."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(model_name in m for m in models)
        except Exception:
            return False
