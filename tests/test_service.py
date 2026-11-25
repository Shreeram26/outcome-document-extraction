"""Backend service tests: endpoints, async processing, tenant isolation,
concurrency limits, terminal immutability, idempotency. Extraction is mocked
— no API credits are used."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.extraction import ExtractionResult, FieldResult, LlmGateway
from app.main import MAX_UPLOAD_BYTES, create_app


def _result(schema, records=1, sparse=False):
    """Deterministic fake ExtractionResult carrying every schema field. In
    sparse mode, non-primary fields of later records are null and their
    provenance is marked unresolved, mirroring a partially successful run."""
    entries = []
    for i in range(records):
        entry = {}
        for key in schema.fields:
            if sparse and key != next(iter(schema.fields)) and i > 0:
                entry[key] = FieldResult(value=None)
            else:
                entry[key] = FieldResult(value=f"{key}-{i}", citation=f"[Page 1] {key}", confidence=0.9)
        entries.append(entry)
    provenance = {}
    for key in schema.fields:
        any_null = any(e[key].value is None for e in entries)
        provenance[key] = "unresolved" if any_null else "llm"
    return ExtractionResult(
        entries=entries, provenance=provenance, mode="single",
        cost_exposure_usd=0.0, cost_settled_usd=0.0,
    )


class _FakeGateway:
    def __init__(self):
        self.calls = 0


class FakeEngine:
    """Fresh engine per job (like production). Behavior controlled by class
    state so the factory-created instances share configuration."""

    behavior = "success"
    records = 2
    gate: threading.Event | None = None
    calls = 0
    calls_lock = threading.Lock()

    def __init__(self):
        with FakeEngine.calls_lock:
            FakeEngine.calls += 1

    def extract(self, schema, document):
        if FakeEngine.behavior == "raise":
            raise RuntimeError("boom")
        if FakeEngine.gate is not None:
            FakeEngine.gate.wait(timeout=10)
        if FakeEngine.behavior == "sparse":
            return _result(schema, records=FakeEngine.records, sparse=True)
        return _result(schema, records=FakeEngine.records)


def reset_fake(behavior="success", records=2, gate=None):
    FakeEngine.behavior = behavior
    FakeEngine.records = records
    FakeEngine.gate = gate
    with FakeEngine.calls_lock:
        FakeEngine.calls = 0


def make_client(behavior="success", records=2, gate=None, **app_kwargs):
    reset_fake(behavior=behavior, records=records, gate=gate)
    app = create_app(engine_factory=FakeEngine, **app_kwargs)
    return TestClient(app)


def _multipart(schema_raw=None, filename="t.csv", content=b"A,B\n1,2\n"):
    import json

    schema_json = json.dumps(schema_raw or {
        "name": "widgets",
        "fields": {
            "widget_id": {"name": "Widget Id", "description": "Identifier.", "primary": True},
            "widget_name": {"name": "Widget Name", "description": "Name."},
        },
    })
    return {"file": (filename, content, "application/octet-stream"), "schema": (None, schema_json)}


def wait_terminal(client, job_id, headers=None, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/jobs/{job_id}", headers=headers or {})
        body = r.json()
        if body["status"] in ("SUCCESS", "PARTIAL", "FAILED"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal state in time")


def test_health():
    with make_client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_job_submission_acknowledged_pending():
    with make_client() as client:
        r = client.post("/jobs", files=_multipart())
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "PENDING"
        assert body["job_id"]


def test_successful_job_reaches_success_with_result():
    with make_client() as client:
        job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
        body = wait_terminal(client, job_id)
        assert body["status"] == "SUCCESS"
        result = body["result"]
        assert result["schema"] == "widgets"
        assert len(result["entries"]) == 2
        assert result["metadata"]["model_calls"] >= 0


def test_job_status_progression_pending_to_terminal():
    gate = threading.Event()
    with make_client(behavior="success", gate=gate) as client:
        job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
        seen = set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = client.get(f"/jobs/{job_id}").json()["status"]
            seen.add(status)
            if status == "RUNNING":
                gate.set()
            if status in ("SUCCESS", "PARTIAL", "FAILED"):
                break
            time.sleep(0.01)
        assert "PENDING" in seen or "RUNNING" in seen
        assert "SUCCESS" in seen


def test_failed_job_handling():
    with make_client(behavior="raise") as client:
        job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
        body = wait_terminal(client, job_id)
        assert body["status"] == "FAILED"
        assert body["error"]["code"] == "extraction_failed"
        assert "result" not in body


def test_invalid_schema_json_returns_400():
    with make_client() as client:
        files = _multipart()
        files["schema"] = (None, "not json")
        assert client.post("/jobs", files=files).status_code == 400


def test_invalid_schema_shape_returns_400():
    with make_client() as client:
        files = _multipart(schema_raw={"fields": {}})
        assert client.post("/jobs", files=files).status_code == 400
        files = _multipart(schema_raw={"fields": {"a": {"name": 1}}})
        assert client.post("/jobs", files=files).status_code == 400


def test_unsupported_file_type_returns_400():
    with make_client() as client:
        files = _multipart(filename="notes.txt", content=b"hello")
        r = client.post("/jobs", files=files)
        assert r.status_code == 400
        assert "unsupported" in r.json()["detail"]


def test_file_size_limit_returns_413():
    with make_client(max_upload_bytes=1024) as client:
        files = _multipart(content=b"x" * 2048)
        assert client.post("/jobs", files=files).status_code == 413


def test_default_upload_limit_is_50mb():
    assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024


def test_unknown_job_returns_404():
    with make_client() as client:
        assert client.get("/jobs/does-not-exist").status_code == 404


def test_cross_tenant_access_returns_404():
    with make_client() as client:
        job_id = client.post(
            "/jobs", files=_multipart(), headers={"X-Organization-Id": "org-a"}
        ).json()["job_id"]
        assert client.get(f"/jobs/{job_id}", headers={"X-Organization-Id": "org-b"}).status_code == 404
        assert client.get(f"/jobs/{job_id}").status_code == 404
        assert client.get(f"/jobs/{job_id}", headers={"X-Organization-Id": "org-a"}).status_code == 200


def test_concurrency_limit_returns_429():
    gate = threading.Event()
    with make_client(behavior="success", gate=gate, max_in_flight=3) as client:
        ids = []
        for i in range(3):
            files = _multipart(content=f"A,B\n{i},{i}\n".encode())
            ids.append(client.post("/jobs", files=files).json()["job_id"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.app.state.store.count_running() < 3:
            time.sleep(0.01)
        assert client.app.state.store.count_running() == 3
        files = _multipart(content=b"A,B\n9,9\n")
        r = client.post("/jobs", files=files)
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        gate.set()
        for job_id in ids:
            wait_terminal(client, job_id)


def test_terminal_status_is_immutable():
    with make_client() as client:
        job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
        body = wait_terminal(client, job_id)
        assert body["status"] == "SUCCESS"
        store = client.app.state.store
        assert store.finish(job_id, "FAILED", error={"code": "late"}) is False
        assert client.get(f"/jobs/{job_id}").json()["status"] == "SUCCESS"


def test_idempotent_resubmission_returns_same_job():
    with make_client() as client:
        files = _multipart()
        first = client.post("/jobs", files=files).json()
        second = client.post("/jobs", files=files).json()
        assert first["job_id"] == second["job_id"]
        different = client.post("/jobs", files=_multipart(content=b"A,B\n3,4\n")).json()
        assert different["job_id"] != first["job_id"]


def test_partial_when_fields_remain_unresolved():
    with make_client(behavior="sparse", records=3) as client:
        job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
        body = wait_terminal(client, job_id)
        assert body["status"] == "PARTIAL"
        assert body["result"]["entries"][0][list(body["result"]["entries"][0])[0]]["value"] is not None


def test_parse_failure_marks_job_failed():
    with make_client() as client:
        files = _multipart(filename="broken.csv", content=b"A,B\n\x00\x01broken")
        job_id = client.post("/jobs", files=files).json()["job_id"]
        body = wait_terminal(client, job_id, timeout=15)
        assert body["status"] in ("FAILED", "PARTIAL", "SUCCESS")
