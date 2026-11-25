"""Background job processing: queue + bounded worker pool wired to the
existing ExtractionEngine. Extraction logic is not duplicated — the worker
only handles parsing input, calling the engine, contract validation and
job-state transitions."""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .parsing import parse_file
from .extraction import ExtractionEngine, LlmError, normalize_schema
from .evaluation.serialize import result_to_contract
from .store import JobStore

MAX_JOB_SECONDS = 1800


class JobWorker:
    """Consumes accepted job ids from a queue and processes them with a bounded
    pool. `max_workers` is also the maximum number of concurrently RUNNING
    jobs; submissions beyond capacity are refused with 429 by the API layer."""

    def __init__(
        self,
        store: JobStore,
        engine_factory: Callable[[], ExtractionEngine],
        max_workers: int = 10,
        max_job_seconds: int = MAX_JOB_SECONDS,
    ):
        self.store = store
        self.engine_factory = engine_factory
        self.max_job_seconds = max_job_seconds
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self, dispatchers: int = 1):
        for _ in range(dispatchers):
            t = threading.Thread(target=self._dispatch_loop, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()

    def submit(self, job_id: str):
        self._queue.put(job_id)

    def _dispatch_loop(self):
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._pool.submit(self._process, job_id)

    def _process(self, job_id: str):
        job = self.store.get_any(job_id)
        if job is None or job.status != "PENDING":
            return
        started = time.monotonic()
        self.store.set_running(job_id, stage="PARSING")
        try:
            contract, unresolved = self._extract(job)
        except Exception as exc:  # noqa: BLE001 — the job must never crash the worker
            self.store.finish(
                job_id,
                "FAILED",
                error={"code": "extraction_failed", "message": str(exc)[:500]},
            )
            return
        if time.monotonic() - started > self.max_job_seconds:
            self.store.finish(
                job_id,
                "FAILED",
                error={"code": "timeout", "message": "processing exceeded 30 minutes"},
            )
            return
        errors = validate_contract(contract, job.schema_raw)
        if errors:
            self.store.finish(
                job_id,
                "FAILED",
                error={
                    "code": "contract_violation",
                    "message": "result failed output contract validation",
                    "details": errors[:10],
                },
            )
            return
        if not contract["entries"]:
            self.store.finish(
                job_id,
                "FAILED",
                error={"code": "empty_result", "message": "no records extracted"},
            )
            return
        status = "SUCCESS" if unresolved == 0 else "PARTIAL"
        self.store.finish(job_id, status, result=contract)

    def _extract(self, job) -> tuple[dict, int]:
        engine = self.engine_factory()
        schema = normalize_schema(job.schema_raw)
        self.store.set_stage(job.job_id, "EXTRACTING")
        suffix = Path(job.filename).suffix or ".bin"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(job.file_bytes)
            document = parse_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        result = engine.extract(schema, document)
        contract = result_to_contract(result, job.schema_name, latency_ms=0)
        contract["document"] = Path(job.filename).stem
        unresolved = sum(1 for v in result.provenance.values() if v == "unresolved")
        return contract, unresolved


def validate_contract(contract: dict, schema_raw: dict) -> list[str]:
    """Contract check via the assignment's own validator (tools/validate.py)."""
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools import validate

    return validate.validate_contract(contract, schema_raw)
