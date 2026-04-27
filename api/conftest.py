import io
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from schemas import (
    AnalysisReport, ReportSummary, SkeletonSequence,
    Skeleton3D, Keypoint3D, KEYPOINT_NAMES,
)
from api.main import app
from api.storage.job_store import JobStore, JobStatus


def _make_fake_report(video_path: str = "/fake/video.mp4") -> AnalysisReport:
    kps = {n: Keypoint3D(x=0.0, y=0.0, z=0.0, confidence=0.9) for n in KEYPOINT_NAMES}
    skeleton = Skeleton3D(
        frame_idx=0,
        timestamp_s=0.0,
        keypoints=kps,
        scale_px_to_m=0.01,
        person_height_cm=170.0,
    )
    return AnalysisReport(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        video_path=video_path,
        duration_s=2.0,
        total_frames=50,
        analyzed_frames=50,
        person_height_cm=170.0,
        methods_used=["REBA"],
        frame_scores=[],
        summary=ReportSummary(max_reba_score=5, pct_frames_high_risk=0.1),
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_report():
    return _make_fake_report()


@pytest.fixture
def mock_orchestrator(fake_report):
    """Patches AnalysisOrchestrator.run to return a fake report instantly."""
    with patch("api.routes.analysis._orchestrator") as mock:
        mock.run.return_value = fake_report
        yield mock


@pytest.fixture
def mock_rule_extractor():
    """Patches RuleExtractor.extract to return an empty list."""
    with patch("api.routes.rules._extractor") as mock:
        mock.extract.return_value = []
        yield mock


@pytest.fixture
def synthetic_video_bytes(tmp_path) -> bytes:
    """Returns bytes of a valid MP4 video for upload tests."""
    path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (640, 480))
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (300, 50), (340, 430), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        return f.read()
