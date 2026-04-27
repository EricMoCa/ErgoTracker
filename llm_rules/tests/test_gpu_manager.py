"""
Tests para GPUManager.
Los tests normales usan mocks; los de integración real están marcados con @pytest.mark.integration.
"""
import pytest
import requests

from llm_rules.gpu_manager import GPUManager


# ---------------------------------------------------------------------------
# Helpers / fixtures locales
# ---------------------------------------------------------------------------


class MockResponse200:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"models": [{"name": "gemma3:4b"}, {"name": "llama3:8b"}]}


class MockResponseError:
    status_code = 500

    def raise_for_status(self) -> None:
        raise requests.HTTPError("500 Server Error")

    def json(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_release_llm_vram_success(monkeypatch):
    """release_llm_vram no lanza excepción aunque Ollama falle (solo loguea warning)."""
    def mock_post(*args, **kwargs):
        return MockResponse200()

    monkeypatch.setattr("requests.post", mock_post)
    # No debe lanzar excepción
    GPUManager.release_llm_vram("gemma3:4b")


def test_release_llm_vram_connection_error(monkeypatch):
    """release_llm_vram captura errores de conexión y no lanza excepción."""
    def mock_post(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("requests.post", mock_post)
    # No debe lanzar excepción — solo loguea warning
    GPUManager.release_llm_vram("gemma3:4b")


def test_release_llm_vram_http_error(monkeypatch):
    """release_llm_vram captura HTTPError y no lanza excepción."""
    def mock_post(*args, **kwargs):
        return MockResponseError()

    monkeypatch.setattr("requests.post", mock_post)
    GPUManager.release_llm_vram("gemma3:4b")


def test_is_ollama_running_true(monkeypatch):
    """is_ollama_running retorna True cuando Ollama responde 200."""
    def mock_get(*args, **kwargs):
        return MockResponse200()

    monkeypatch.setattr("requests.get", mock_get)
    assert GPUManager.is_ollama_running() is True


def test_is_ollama_running_false_connection_error(monkeypatch):
    """is_ollama_running retorna False cuando Ollama no responde."""
    def mock_get(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("requests.get", mock_get)
    assert GPUManager.is_ollama_running() is False


def test_is_model_available_found(monkeypatch):
    """is_model_available retorna True cuando el modelo está en la lista."""
    def mock_get(*args, **kwargs):
        return MockResponse200()

    monkeypatch.setattr("requests.get", mock_get)
    assert GPUManager.is_model_available("gemma3:4b") is True


def test_is_model_available_not_found(monkeypatch):
    """is_model_available retorna False cuando el modelo no está en la lista."""
    def mock_get(*args, **kwargs):
        return MockResponse200()

    monkeypatch.setattr("requests.get", mock_get)
    assert GPUManager.is_model_available("nonexistent_model:99b") is False


def test_is_model_available_connection_error(monkeypatch):
    """is_model_available retorna False cuando Ollama no responde."""
    def mock_get(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("requests.get", mock_get)
    assert GPUManager.is_model_available("gemma3:4b") is False


@pytest.mark.integration
def test_is_ollama_running_real():
    """[INTEGRACIÓN] Verifica si Ollama real está corriendo."""
    result = GPUManager.is_ollama_running()
    # Solo verificamos que retorna un bool — no forzamos que esté activo
    assert isinstance(result, bool)


@pytest.mark.integration
def test_release_llm_vram_real():
    """[INTEGRACIÓN] Llama release con Ollama real — no debe lanzar excepción."""
    GPUManager.release_llm_vram("gemma3:4b")
