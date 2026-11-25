"""In-memory job store with tenant isolation and terminal-state immutability."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

TERMINAL_STATUSES = {"SUCCESS", "PARTIAL", "FAILED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    org: str
    filename: str = ""
    file_bytes: bytes = b""
    schema_raw: dict = field(default_factory=dict)
    schema_name: str = ""
    webhook_url: str | None = None
    status: str = "PENDING"
    stage: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    result: dict | None = None
    error: dict | None = None


class JobStore:
    """Thread-safe in-memory job registry.

    - `get` enforces tenant isolation: a job is visible only to its own
      organization; unknown ids and foreign jobs are indistinguishable (None).
    - Terminal statuses (SUCCESS / PARTIAL / FAILED) are immutable: `finish`
      is ignored once a job reached a terminal state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._idem: dict[str, str] = {}

    def create(
        self,
        org: str,
        filename: str,
        file_bytes: bytes,
        schema_raw: dict,
        schema_name: str,
        idem_key: str = "",
        webhook_url: str | None = None,
    ) -> tuple[Job, bool]:
        """Create a job, or return the existing one for an idempotency key.
        Returns (job, created)."""
        with self._lock:
            if idem_key and idem_key in self._idem:
                return self._jobs[self._idem[idem_key]], False
            job = Job(
                job_id=uuid.uuid4().hex,
                org=org,
                filename=filename,
                file_bytes=file_bytes,
                schema_raw=schema_raw,
                schema_name=schema_name,
                webhook_url=webhook_url,
            )
            self._jobs[job.job_id] = job
            if idem_key:
                self._idem[idem_key] = job.job_id
            return job, True

    def get(self, job_id: str, org: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.org != org:
                return None
            return job

    def get_by_idem(self, idem_key: str) -> Job | None:
        with self._lock:
            job_id = self._idem.get(idem_key)
            return self._jobs.get(job_id) if job_id else None

    def get_any(self, job_id: str) -> Job | None:
        """Internal (worker) access without tenant scoping."""
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str, stage: str | None = None) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.status = "RUNNING"
            job.stage = stage or job.stage
            job.updated_at = _now()
            return True

    def set_stage(self, job_id: str, stage: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.stage = stage
            job.updated_at = _now()
            return True

    def finish(
        self,
        job_id: str,
        status: str,
        result: dict | None = None,
        error: dict | None = None,
    ) -> bool:
        """Move a job to a terminal state. Ignored if the job is already
        terminal — terminal statuses never change."""
        if status not in TERMINAL_STATUSES:
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.status = status
            job.result = result
            job.error = error
            job.stage = None
            job.updated_at = _now()
            return True

    def count_running(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "RUNNING")
