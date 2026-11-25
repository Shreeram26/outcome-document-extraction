"""End-to-end evaluation harness: run the extraction pipeline over the sample
corpus, compare against the provided ground truth with the assignment's own
validator (tools/validate.py), and emit a machine-readable JSON report.

    python -m app.evaluation.harness --model google/gemini-3.8-flash

The harness never contains model credentials: the gateway reads LLM_BASE_URL /
LLM_API_KEY from the environment (loaded from .env by the CLI)."""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.extraction.engine import ExtractionEngine
from app.extraction.llm import LlmError, LlmGateway
from app.extraction.schema import normalize_schema
from app.parsing import parse_file
from .serialize import result_to_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import validate  # noqa: E402  (the assignment's own validator)

DEFAULT_MODEL = "google/gemini-3.8-flash"
SAMPLE_DOCS_DIR = REPO_ROOT / "sample" / "sample-data-documents"
SAMPLE_SCHEMAS_DIR = REPO_ROOT / "sample" / "schemas"
SAMPLE_TRUTH_DIR = REPO_ROOT / "sample" / "sample-data-output-validated"
MANIFEST_PATH = REPO_ROOT / "sample" / "manifest.csv"


@dataclass
class CorpusItem:
    document_id: str
    file: Path
    schema_name: str
    schema_path: Path
    truth_path: Path | None


def load_manifest(
    manifest_path: Path = MANIFEST_PATH,
    docs_dir: Path = SAMPLE_DOCS_DIR,
    schemas_dir: Path = SAMPLE_SCHEMAS_DIR,
    truth_dir: Path = SAMPLE_TRUTH_DIR,
) -> list[CorpusItem]:
    """Sample corpus from the provided manifest, schemas and truth directory."""
    items: list[CorpusItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            doc_id = row["document_id"].strip()
            items.append(
                CorpusItem(
                    document_id=doc_id,
                    file=docs_dir / row["file"].strip(),
                    schema_name=row["schema_name"].strip(),
                    schema_path=schemas_dir / f"{row['schema_name'].strip()}.json",
                    truth_path=truth_dir / f"{doc_id}.json" if (truth_dir / f"{doc_id}.json").exists() else None,
                )
            )
    return items


def evaluate_document(
    engine: ExtractionEngine, item: CorpusItem
) -> tuple[dict, dict]:
    """Extract one document, validate against the contract, score against truth.
    Returns (contract_result, document_report)."""
    schema_raw = json.loads(item.schema_path.read_text(encoding="utf-8"))
    schema = normalize_schema(schema_raw)
    document = parse_file(item.file)
    started = time.monotonic()
    result = engine.extract(schema, document)
    latency_ms = int((time.monotonic() - started) * 1000)
    contract = result_to_contract(result, schema_raw.get("name", ""), latency_ms)
    contract["document"] = item.document_id
    contract_errors = validate.validate_contract(contract, schema_raw)
    report: dict = {
        "document": item.document_id,
        "schema": item.schema_name,
        "mode": result.mode,
        "latency_ms": latency_ms,
        "model_calls": contract["metadata"]["model_calls"],
        "tokens_in": contract["metadata"]["tokens_in"],
        "tokens_out": contract["metadata"]["tokens_out"],
        "cost_usd": contract["metadata"]["cost_usd"],
        "cost_settled_usd": contract["metadata"].get("cost_settled_usd", 0.0),
        "call_log": list(result.call_log),
        "conflicts": list(result.conflicts),
        "contract_valid": not contract_errors,
        "contract_errors": contract_errors,
        "warnings": list(result.warnings),
        "field_provenance": _provenance_keys(result),
        "entries": len(result.entries),
    }
    if item.truth_path is not None:
        truth = json.loads(item.truth_path.read_text(encoding="utf-8"))
        scored = validate.score(contract, truth, schema_raw)
        report.update(_score_section(scored, len(schema.fields), len(result.entries)))
    return contract, report


def evaluate_corpus(
    engine: ExtractionEngine, items: list[CorpusItem], write_outputs_dir: Path | None = None
) -> dict:
    """Evaluate the corpus; optionally write per-document contract JSON files."""
    documents = []
    for item in items:
        contract, report = evaluate_document(engine, item)
        if write_outputs_dir is not None:
            write_outputs_dir.mkdir(parents=True, exist_ok=True)
            out = write_outputs_dir / f"{item.document_id}.json"
            out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
            report["output_file"] = str(out)
        documents.append(report)
        status = "VALID" if report["contract_valid"] else "INVALID"
        line = f"{status:7} {item.document_id:16} mode={report['mode']:6}"
        if "validator_score" in report:
            line += f" score={report['validator_score']:.3f}"
        line += f" tokens={report['tokens_in']}/{report['tokens_out']} {report['latency_ms']}ms"
        print(line, flush=True)
    return {"documents": documents, "overall": _overall(documents)}


def _provenance_keys(result) -> dict:
    keys = {"deterministic": [], "llm": [], "unresolved": []}
    for key, source in result.provenance.items():
        keys.setdefault(source, []).append(key)
    return keys


def _score_section(scored: dict, n_fields: int, n_entries: int) -> dict:
    total_cells = max(n_fields * max(n_entries, 1), 1)
    incorrect_cells = sum(round((1.0 - v) * n_entries) for v in scored["per_field"].values())
    truth_rows = scored["rows"]["truth"]
    row_accuracy = (
        (truth_rows - len(scored["missing_rows"])) / truth_rows if truth_rows else 1.0
    )
    return {
        "validator_score": scored["score"],
        "field_accuracy": scored["per_field"],
        "field_accuracy_mean": scored["score"],
        "incorrect_cells": incorrect_cells,
        "total_cells": total_cells,
        "hallucinated_cells": len(scored["hallucinations"]),
        "missed_cells": len(scored["misses"]),
        "hallucinated_fields": scored["hallucinations"],
        "missed_fields": scored["misses"],
        "extra_rows": scored["extra_rows"],
        "missing_rows": scored["missing_rows"],
        "rows": scored["rows"],
        "row_accuracy": round(row_accuracy, 4),
    }


def _overall(documents: list[dict]) -> dict:
    scored = [d for d in documents if "validator_score" in d]
    overall: dict = {
        "documents": len(documents),
        "contract_valid": sum(1 for d in documents if d["contract_valid"]),
        "mean_validator_score": (
            sum(d["validator_score"] for d in scored) / len(scored) if scored else None
        ),
        "row_accuracy_mean": (
            sum(d["row_accuracy"] for d in scored) / len(scored) if scored else None
        ),
        "model_calls": sum(d["model_calls"] for d in documents),
        "tokens_in": sum(d["tokens_in"] for d in documents),
        "tokens_out": sum(d["tokens_out"] for d in documents),
        "cost_usd": round(sum(d["cost_usd"] for d in documents), 6),
        "latency_ms_total": sum(d["latency_ms"] for d in documents),
        "hallucinated_cells": sum(d.get("hallucinated_cells", 0) for d in documents),
        "missed_cells": sum(d.get("missed_cells", 0) for d in documents),
        "incorrect_cells": sum(d.get("incorrect_cells", 0) for d in documents),
        "extra_rows": sum(len(d.get("extra_rows", [])) for d in documents),
        "missing_rows": sum(len(d.get("missing_rows", [])) for d in documents),
    }
    overall["provenance"] = _provenance_accuracy(documents)
    overall["worst_fields"] = _worst_fields(documents)
    return overall


def _provenance_accuracy(documents: list[dict]) -> dict:
    """Mean validator field accuracy for fields resolved deterministically vs by
    the LLM, over all (document, field) pairs where the field was scored."""
    buckets: dict[str, list[float]] = {"deterministic": [], "llm": [], "unresolved": []}
    for d in documents:
        provenance = d.get("field_provenance", {})
        accuracy = d.get("field_accuracy", {})
        for source, keys in provenance.items():
            for key in keys:
                if key in accuracy:
                    buckets.setdefault(source, []).append(accuracy[key])
    return {
        source: (sum(v) / len(v)) if v else None
        for source, v in buckets.items()
    }


def _worst_fields(documents: list[dict], limit: int = 15) -> list[dict]:
    rows = []
    for d in documents:
        for field_key, acc in d.get("field_accuracy", {}).items():
            if acc < 1.0:
                rows.append(
                    {
                        "document": d["document"],
                        "field": field_key,
                        "accuracy": acc,
                        "provenance": next(
                            (
                                src
                                for src, keys in d.get("field_provenance", {}).items()
                                if field_key in keys
                            ),
                            "unknown",
                        ),
                    }
                )
    rows.sort(key=lambda r: r["accuracy"])
    return rows[:limit]


def select_items(items: list[CorpusItem], document_ids: list[str] | None) -> list[CorpusItem]:
    """Filter the corpus to the requested document ids (all when None)."""
    if not document_ids:
        return list(items)
    wanted = {d.strip() for d in document_ids}
    missing = wanted - {i.document_id for i in items}
    if missing:
        raise ValueError(f"unknown document id(s): {sorted(missing)}")
    return [i for i in items if i.document_id in wanted]


def build_engine(
    model: str,
    skip_llm: bool = False,
    dry_run: bool = False,
    chunk_chars: int | None = None,
    max_calls: int | None = None,
    max_cost_usd: float | None = None,
    run_cost_cap: float | None = None,
    max_tokens: int | None = None,
) -> ExtractionEngine:
    """Build an engine whose gateway shares one CostLedger enforcing the
    run-level cap (exposure = settled + retained reservations)."""
    from types import SimpleNamespace

    from app.extraction.ledger import CostLedger

    engine_kwargs: dict = {}
    if chunk_chars is not None:
        engine_kwargs["chunk_chars"] = chunk_chars
    if max_calls is not None:
        engine_kwargs["max_calls"] = max_calls
    if max_cost_usd is not None:
        engine_kwargs["max_cost_usd"] = max_cost_usd
    gateway_kwargs: dict = {"ledger": CostLedger(cap_usd=run_cost_cap) if run_cost_cap else None}
    if max_tokens is not None:
        gateway_kwargs["default_max_tokens"] = max_tokens
    if skip_llm:
        class _NoLlm:
            def create(self, **_kwargs):  # pragma: no cover - trivial stub
                raise LlmError("LLM disabled (--skip-llm)")

        transport = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_NoLlm().create))
        )
        return ExtractionEngine(LlmGateway(transport=transport, **gateway_kwargs), model, **engine_kwargs)
    if dry_run:
        dry_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"entries": []}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, cost=0.0),
        )

        class _DryRun:
            def create(self, **_kwargs):
                return dry_response

        transport = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_DryRun().create))
        )
        return ExtractionEngine(LlmGateway(transport=transport, **gateway_kwargs), model, **engine_kwargs)
    return ExtractionEngine(LlmGateway(**gateway_kwargs), model, **engine_kwargs)


def main(argv=None) -> int:
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--documents", nargs="*", default=None, help="document ids; default all")
    ap.add_argument("--report", default=str(REPO_ROOT / "outputs" / "eval-report.json"))
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--skip-llm", action="store_true", help="deterministic stage only")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="plan LLM calls without any network usage or token spend",
    )
    ap.add_argument("--max-calls", type=int, default=None, help="max LLM calls per document")
    ap.add_argument("--max-cost", type=float, default=None, help="max cumulative cost per document (USD)")
    ap.add_argument("--chunk-chars", type=int, default=None, help="max context characters per call")
    ap.add_argument("--max-tokens", type=int, default=None, help="max output tokens per LLM call (truncation guard)")
    ap.add_argument(
        "--run-cost-cap", type=float, default=None,
        help="hard cumulative spend cap for the whole run (USD, provider-reported)",
    )
    args = ap.parse_args(argv)

    items = load_manifest()
    try:
        items = select_items(items, args.documents)
    except ValueError as exc:
        ap.error(str(exc))
    if not items:
        ap.error("no documents selected (check sample/manifest.csv)")
    engine = build_engine(
        args.model,
        skip_llm=args.skip_llm,
        dry_run=args.dry_run,
        chunk_chars=args.chunk_chars,
        max_calls=args.max_calls,
        max_cost_usd=args.max_cost,
        max_tokens=args.max_tokens,
        run_cost_cap=args.run_cost_cap,
    )
    outputs_dir = None if args.dry_run else Path(args.outputs_dir)
    report = evaluate_corpus(engine, items, write_outputs_dir=outputs_dir)
    run_cap_status = engine.gateway.ledger.status() if engine.gateway.ledger else None
    full = {
        "model": None if args.skip_llm or args.dry_run else args.model,
        "skip_llm": args.skip_llm,
        "dry_run": args.dry_run,
        "run_cost_cap": run_cap_status,
        "documents": report["documents"],
        "overall": report["overall"],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    overall = report["overall"]
    print(
        f"\nmean score={overall['mean_validator_score']:.3f}"
        f" row_accuracy={overall['row_accuracy_mean']:.3f}"
        f" contract_valid={overall['contract_valid']}/{overall['documents']}"
        f" calls={overall['model_calls']}"
        f" tokens={overall['tokens_in']}/{overall['tokens_out']}"
        f" cost=${overall['cost_usd']:.4f}"
    )
    print(f"report: {report_path}")
    if run_cap_status:
        print(f"run cost cap (ledger): {run_cap_status}")
    return 0 if overall["contract_valid"] == overall["documents"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
