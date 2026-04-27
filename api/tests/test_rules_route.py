import io
import pytest


def test_extract_rules_returns_202(client, mock_rule_extractor):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/rules/extract",
        files={"pdf": ("rules.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"profile_name": "iso_11228"},
    )
    assert response.status_code == 202


def test_extract_rules_returns_job_id(client, mock_rule_extractor):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/rules/extract",
        files={"pdf": ("rules.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={},
    )
    data = response.json()
    assert "job_id" in data
    assert data["job_id"]


def test_extract_rules_default_profile_name(client, mock_rule_extractor):
    pdf_bytes = b"%PDF-1.4 fake pdf"
    response = client.post(
        "/rules/extract",
        files={"pdf": ("rules.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 202


def test_get_extraction_status_not_found(client):
    response = client.get("/rules/nonexistent-id")
    assert response.status_code == 404


def test_get_extraction_status_returns_status(client, mock_rule_extractor):
    pdf_bytes = b"%PDF-1.4 fake"
    create_response = client.post(
        "/rules/extract",
        files={"pdf": ("r.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={},
    )
    job_id = create_response.json()["job_id"]

    status_response = client.get(f"/rules/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert "status" in data
    assert data["job_id"] == job_id


def test_get_extraction_status_has_error_field(client, mock_rule_extractor):
    pdf_bytes = b"%PDF-1.4 fake"
    create_response = client.post(
        "/rules/extract",
        files={"pdf": ("r.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={},
    )
    job_id = create_response.json()["job_id"]

    status_response = client.get(f"/rules/{job_id}")
    data = status_response.json()
    assert "error" in data
