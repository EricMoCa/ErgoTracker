import io
import pytest
from api.storage.job_store import JobStatus


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_has_version(client):
    response = client.get("/health")
    assert "version" in response.json()


def test_start_analysis_returns_202(client, mock_orchestrator, synthetic_video_bytes):
    response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={"person_height_cm": "170.0", "ergo_methods": "REBA"},
    )
    assert response.status_code == 202


def test_start_analysis_returns_job_id(client, mock_orchestrator, synthetic_video_bytes):
    response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={"person_height_cm": "170.0"},
    )
    data = response.json()
    assert "job_id" in data
    assert data["job_id"]


def test_poll_job_status_not_found(client):
    response = client.get("/analyze/nonexistent-job-id")
    assert response.status_code == 404


def test_poll_job_status_returns_status(client, mock_orchestrator, synthetic_video_bytes):
    create_response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={},
    )
    job_id = create_response.json()["job_id"]

    poll_response = client.get(f"/analyze/{job_id}")
    assert poll_response.status_code == 200
    data = poll_response.json()
    assert "status" in data
    assert "job_id" in data


def test_poll_job_status_has_correct_id(client, mock_orchestrator, synthetic_video_bytes):
    create_response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={},
    )
    job_id = create_response.json()["job_id"]

    poll_response = client.get(f"/analyze/{job_id}")
    assert poll_response.json()["job_id"] == job_id


def test_start_analysis_multiple_methods(client, mock_orchestrator, synthetic_video_bytes):
    response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={"ergo_methods": "REBA,RULA,OWAS"},
    )
    assert response.status_code == 202


def test_start_analysis_custom_height(client, mock_orchestrator, synthetic_video_bytes):
    response = client.post(
        "/analyze/",
        files={"video": ("test.mp4", io.BytesIO(synthetic_video_bytes), "video/mp4")},
        data={"person_height_cm": "180.0"},
    )
    assert response.status_code == 202
