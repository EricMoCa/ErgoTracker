import pytest
from pathlib import Path


def test_download_report_not_found(client):
    response = client.get("/reports/nonexistent-id/download")
    assert response.status_code == 404


def test_download_report_returns_pdf(client, tmp_path, monkeypatch):
    """When report PDF exists on disk, endpoint returns it."""
    report_id = "test-report-123"
    pdf_content = b"%PDF-1.4 fake pdf"

    # Patch settings.reports_output_dir to tmp_path
    from api import config as api_config
    monkeypatch.setattr(api_config.settings, "reports_output_dir", tmp_path)

    # Also patch the router's reference to settings
    import api.routes.reports as reports_module
    monkeypatch.setattr(reports_module.settings, "reports_output_dir", tmp_path)

    pdf_file = tmp_path / f"{report_id}.pdf"
    pdf_file.write_bytes(pdf_content)

    response = client.get(f"/reports/{report_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_download_report_filename_contains_id(client, tmp_path, monkeypatch):
    report_id = "abc-456"
    import api.routes.reports as reports_module
    monkeypatch.setattr(reports_module.settings, "reports_output_dir", tmp_path)

    pdf_file = tmp_path / f"{report_id}.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    response = client.get(f"/reports/{report_id}/download")
    assert report_id in response.headers.get("content-disposition", "")


def test_api_has_analysis_routes(client):
    """OpenAPI schema includes /analyze routes."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert any("/analyze" in p for p in paths)


def test_api_has_rules_routes(client):
    response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert any("/rules" in p for p in paths)


def test_api_has_reports_routes(client):
    response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert any("/reports" in p for p in paths)
