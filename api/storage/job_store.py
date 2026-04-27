from enum import Enum
from datetime import datetime
from typing import Any
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_type: str, params: dict):
        self.id = str(uuid.uuid4())
        self.job_type = job_type
        self.params = params
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.result: Any = None
        self.error: str | None = None


class JobStore:
    """
    In-memory store for analysis jobs.
    Jobs are lost on server restart — no database needed for v1.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create(self, job_type: str, params: dict) -> Job:
        job = Job(job_type, params)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def list_all(self) -> list[Job]:
        return list(self._jobs.values())
