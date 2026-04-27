"""
llm_rules/conftest.py
Fixtures compartidas para tests de llm_rules.
"""
import json

import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    """Crea un PDF mínimo con texto ergonómico sintético para tests."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        (
            "Artículo 4.2 - Posturas de trabajo\n"
            "La flexión del tronco no debe superar los 45 grados durante más de 2 horas.\n"
            "Los ángulos de elevación del hombro superiores a 60 grados representan\n"
            "riesgo alto y deben evitarse. La flexión del cuello debe mantenerse por\n"
            "debajo de 20 grados durante operaciones prolongadas."
        ),
    )
    path = tmp_path / "test_normativa.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock de Ollama para tests sin servidor real."""

    def mock_post(*args, **kwargs):
        class MockResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "id": "R-001",
                                    "description": "Flexión de tronco máx 45°",
                                    "joint": "trunk_flexion",
                                    "condition": "angle > 45",
                                    "risk_level": "HIGH",
                                    "action": "Reducir ángulo o rotar al operario",
                                    "source": "PDF:test_normativa.pdf",
                                }
                            ]
                        )
                    }
                }

        return MockResponse()

    monkeypatch.setattr("requests.post", mock_post)
