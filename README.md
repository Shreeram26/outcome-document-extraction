# Outcome Document Extraction Service

AI-powered document extraction service for the Outcome AI/ML Developer take-home assignment.

## Validation scores (sample set, `tools/validate.py`)

| Document | Score | Rows | Extraction method |
|---|---|---|---|
| boxscore_001 | 1.000 | 10/10 | deterministic (no LLM) |
| budget_001 | 0.974 | 55/55 | coverage-driven LLM |
| lease_001 | 0.725 | 1/1 | vision (scanned PDF) |
| lease_002 | 0.650 | 1/1 | text LLM + schema defaults |
| lease_003 | 0.825 | 1/1 | vision (scanned PDF) |
| om_001 | 0.862 | 1/1 | text LLM |
| rentroll_001 | 0.831 | 325/325 | deterministic + table fallback |
| t12_001 | 0.670 | 127/119 | grid fallback (row-bounded) |
| **Mean** | **0.817** | | |

Model: `anthropic/claude-haiku-5.5` via OpenRouter. Total API cost for all 8 documents: $0.084.

## How to run

### Docker (assignment contract)

```bash
docker compose up
# GET http://localhost:8080/health -> {"status": "ok"}
```

The service listens on port 8080. LLM credentials are read from the
environment — provide `LLM_API_KEY` (and optionally `LLM_BASE_URL`,
`LLM_MODEL`, `MAX_COST_PER_DOC_USD`, `COST_CAP_USD`) when starting
Compose, e.g. `LLM_API_KEY=... docker compose up`.

### Local (without Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put the real key into .env
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Validation (no API calls needed)

```bash
python tools/validate.py outputs/ --schema-dir sample/schemas --truth-dir sample/sample-data-output-validated
```

All 8 outputs print `VALID`.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## API

- `GET /health` — liveness, returns `{"status": "ok"}`.
- `POST /jobs` — multipart form with `file` (PDF/DOCX/XLSX/CSV, ≤ 50 MB),
  `schema` (JSON string describing the fields to extract) and optional
  `webhook_url`. Returns `202` with `{job_id, status: "PENDING"}` in under
  2 seconds; extraction runs in the background. Optional
  `X-Organization-Id` header scopes the job to a tenant. Duplicate
  submissions of the same document + schema return the existing job.
  `429` when 10 jobs are already in flight; `400` for invalid input;
  `413` above 50 MB.
- `GET /jobs/{job_id}` — `{status, result?, error?}`. `result` carries
  `{entries, metadata}` and is present once the job is `SUCCESS` or
  `PARTIAL`. Terminal statuses are immutable. Unknown jobs and jobs
  belonging to another organization return `404`.

Job statuses: `PENDING → RUNNING → SUCCESS | PARTIAL | FAILED`.

## Architecture

- `app/parsing/` — format parsers (PDF/DOCX/XLSX/CSV) with source locations
- `app/extraction/` — schema normalization, deterministic-first extraction,
  progressive LLM fallback with cost reservations, vision for scanned PDFs
  (see `DESIGN.md`)
- `app/store.py` / `app/worker.py` / `app/main.py` — job store, background
  worker pool (10 in-flight, 30-minute guard), FastAPI endpoints
- `tools/validate.py` — assignment-provided validator; the service validates
  every result against it before marking a job terminal
- Tests: `pytest` (no API credits needed — extraction is mocked at the
  engine boundary; real-gateway validation runs live under `experiments/`)
