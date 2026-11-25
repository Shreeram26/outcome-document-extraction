"""Outcome document extraction service — HTTP API.

Endpoints (assignment contract, contract/openapi.yaml):
- GET  /health          liveness
- POST /jobs            multipart upload (document + schema), async extraction
- GET  /jobs/{job_id}   status / result, tenant-scoped

All LLM traffic goes through the existing ExtractionEngine and the
LLM_BASE_URL / LLM_API_KEY environment variables. The cost ledger and
per-document budgets from the extraction core remain active."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile

from .extraction import LlmGateway, SchemaError, normalize_schema
from .extraction.ledger import CostLedger
from .parsing import SUPPORTED_EXTENSIONS
from .store import JobStore
from .worker import JobWorker

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IN_FLIGHT = 10
DEFAULT_ORG = "default"
DEFAULT_MODEL = "google/gemini-3.8-flash"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def create_default_engine():
    """Fresh engine per job: ExtractionEngine keeps per-document accounting
    state, so concurrent jobs must not share one instance."""
    ledger = None
    cap = os.environ.get("COST_CAP_USD")
    if cap:
        ledger = CostLedger(cap_usd=float(cap))
    gateway = LlmGateway(ledger=ledger)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    return ExtractionEngine(
        gateway,
        model,
        max_cost_usd=_env_float("MAX_COST_PER_DOC_USD", 0.5),
    )


def job_to_response(job) -> dict:
    data = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.stage:
        data["stage"] = job.stage
    if job.status in ("SUCCESS", "PARTIAL") and job.result is not None:
        data["result"] = job.result
    if job.status == "FAILED" and job.error is not None:
        data["error"] = job.error
    return data


def create_app(
    engine_factory=None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_in_flight: int = MAX_IN_FLIGHT,
) -> FastAPI:
    load_dotenv()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Outcome Document Extraction Service", version="1.0.0", lifespan=lifespan)
    store = JobStore()
    worker = JobWorker(store, engine_factory or create_default_engine, max_workers=max_in_flight)
    app.state.store = store
    app.state.worker = worker

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/jobs", status_code=202)
    async def create_job(
        request: Request,
        x_organization_id: str | None = Header(None, alias="X-Organization-Id"),
    ):
        """Multipart form: file (binary), schema (JSON string), optional
        webhook_url. Field names follow contract/openapi.yaml."""
        org = x_organization_id or DEFAULT_ORG

        form = await request.form()
        upload = form.get("file")
        schema_json = form.get("schema")
        webhook_url = form.get("webhook_url")
        if upload is None or schema_json is None:
            raise HTTPException(status_code=400, detail="multipart fields 'file' and 'schema' are required")
        if not isinstance(upload, StarletteUploadFile):
            raise HTTPException(status_code=400, detail="'file' must be a file upload")

        if upload.size is not None and upload.size > max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        raw = await upload.read()
        if len(raw) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        if not raw:
            raise HTTPException(status_code=400, detail="empty file")

        filename = upload.filename or ""
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type {extension!r}; supported:"
                f" {', '.join(SUPPORTED_EXTENSIONS)}",
            )

        try:
            schema_raw = json.loads(str(schema_json))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"schema is not valid JSON: {exc}")
        try:
            normalize_schema(schema_raw)
        except SchemaError as exc:
            raise HTTPException(status_code=400, detail=f"invalid schema: {exc}")

        idem_key = hashlib.sha256(
            "|".join([
                org,
                filename,
                hashlib.sha256(raw).hexdigest(),
                json.dumps(schema_raw, sort_keys=True, default=str),
            ]).encode()
        ).hexdigest()

        duplicate = store.get_by_idem(idem_key)
        if duplicate is not None:
            return job_to_response(duplicate)

        if store.count_running() >= max_in_flight:
            raise HTTPException(
                status_code=429,
                detail="concurrency capacity exceeded; retry later",
                headers={"Retry-After": "5"},
            )
        job, created = store.create(
            org=org,
            filename=filename,
            file_bytes=raw,
            schema_raw=schema_raw,
            schema_name=str(schema_raw.get("name") or extension.lstrip(".")),
            idem_key=idem_key,
            webhook_url=webhook_url,
        )
        worker.submit(job.job_id)
        return job_to_response(job)

    @app.get("/jobs/{job_id}")
    def get_job(
        job_id: str,
        x_organization_id: str | None = Header(None, alias="X-Organization-Id"),
    ):
        org = x_organization_id or DEFAULT_ORG
        job = store.get(job_id, org)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job_to_response(job)

    return app


app = create_app()
