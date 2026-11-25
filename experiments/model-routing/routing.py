"""Model-routing experiments: compare models and structured-output techniques
under hard cost caps. Run sequentially; every call goes through the shared
CostLedger (never bypassed). Results land in this directory as JSON.

Usage:
    python experiments/model-routing/routing.py probe     # capability probes
    python experiments/model-routing/routing.py run       # experiment matrix
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.extraction import (  # noqa: E402
    ExtractionResult,
    FieldResult,
    LlmBadResponseError,
    LlmBudgetError,
    LlmError,
    LlmGateway,
    normalize_schema,
    parse_extraction_response,
)


def _load_env():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
from app.extraction.context import select_context  # noqa: E402
from app.extraction.ledger import CostLedger, usage_cost  # noqa: E402
from app.extraction.prompting import SYSTEM_PROMPT, build_messages  # noqa: E402
from app.evaluation.serialize import result_to_contract  # noqa: E402
from app.parsing import parse_file  # noqa: E402
from tools import validate  # noqa: E402

_load_env()
AGGREGATE_CAP_USD = 1.20          # this session's share of the $1.50 pass cap
PER_EXPERIMENT_CAP_USD = 0.35
SAMPLES = REPO / "sample" / "sample-data-documents"
SCHEMAS = REPO / "sample" / "schemas"
TRUTH = REPO / "sample" / "sample-data-output-validated"
HERE = Path(__file__).resolve().parent

BASELINE_MODEL = "anthropic/claude-haiku-5.5"
CHEAP_MODEL = "openai/gpt-6-luna"
STRONG_MODEL = "anthropic/claude-sonnet-5.5"


def _response_format(schema_raw: dict) -> dict:
    """Provider-safe JSON Schema: no type arrays (Anthropic strict validation
    rejects them) — `value` is left untyped since extractions mix strings,
    numbers, booleans and nulls."""
    fields = schema_raw["fields"]
    props = {
        key: {
            "type": "object",
            "properties": {
                "value": {},
                "citation": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["value", "citation", "confidence"],
            "additionalProperties": False,
        }
        for key in fields
    }
    entry = {"type": "object", "properties": props, "required": list(props),
             "additionalProperties": False}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "schema": {
                "type": "object",
                "properties": {"entries": {"type": "array", "items": entry}},
                "required": ["entries"],
                "additionalProperties": False,
            },
        },
    }


def _tools(schema_raw: dict) -> dict:
    return {
        "tools": [{
            "type": "function",
            "function": {
                "name": "submit_extraction",
                "description": "Submit the extracted records for the document.",
                "parameters": _response_format(schema_raw)["json_schema"]["schema"],
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "submit_extraction"}},
    }


def method_kwargs(method: str, schema_raw: dict) -> dict | None:
    if method == "json_prompt":
        return None
    if method == "json_schema":
        return {"response_format": _response_format(schema_raw)}
    if method == "tools":
        return _tools(schema_raw)
    raise ValueError(method)


def content_of(response_raw, method: str) -> str | None:
    if method != "tools":
        return getattr(response_raw.choices[0].message, "content", None)
    calls = getattr(response_raw.choices[0].message, "tool_calls", None) or []
    for call in calls:
        fn = getattr(call, "function", None)
        if fn is not None and getattr(fn, "name", "") == "submit_extraction":
            return getattr(fn, "arguments", None)
    return None


def probe(gateway: LlmGateway, model: str, method: str) -> dict:
    """Tiny capability probe: does the API accept the method's parameters?"""
    schema_raw = {"name": "probe", "fields": {"ok": {"name": "Ok", "description": "ok"}}}
    extra = method_kwargs(method, schema_raw) if method != "json_prompt" else None
    started = time.monotonic()
    try:
        response = gateway.complete(
            model,
            [{"role": "user", "content": 'Reply with {"ok": true}'}],
            max_tokens=512,
            extra_params=extra,
        )
        content = content_of(response.raw, method)
        return {"model": model, "method": method, "accepted": True,
                "content_kind": "tool_call" if method == "tools" and content is not None else "content",
                "latency_ms": int((time.monotonic() - started) * 1000)}
    except LlmError as exc:
        return {"model": model, "method": method, "accepted": False,
                "error": str(exc)[:200],
                "latency_ms": int((time.monotonic() - started) * 1000)}


class Budget:
    """Aggregate + per-experiment cost accounting on top of the shared ledger."""

    def __init__(self, ledger: CostLedger, aggregate_cap: float, per_experiment_cap: float):
        self.ledger = ledger
        self.aggregate_cap = aggregate_cap
        self.per_experiment_cap = per_experiment_cap
        self.experiment_start = 0.0

    def begin_experiment(self):
        self.experiment_start = self.ledger.settled_usd + self.ledger.outstanding_usd

    def experiment_spend(self) -> float:
        return (self.ledger.settled_usd + self.ledger.outstanding_usd) - self.experiment_start

    def aggregate_spend(self) -> float:
        return self.ledger.settled_usd + self.ledger.outstanding_usd

    def may_call(self) -> bool:
        return (self.aggregate_spend() < self.aggregate_cap
                and self.experiment_spend() < self.per_experiment_cap)


def _extract_content(response_raw, method: str) -> str | None:
    return content_of(response_raw, method)


def run_variant(
    gateway: LlmGateway,
    budget: Budget,
    exp_id: str,
    model: str,
    category: str,
    method: str,
    document_path: Path,
    schema_path: Path,
    truth_path: Path,
    repair: bool = True,
) -> dict:
    """One experiment configuration: single targeted-context call (+ one
    bounded repair), scored against ground truth."""
    schema_raw = json.loads(schema_path.read_text())
    schema = normalize_schema(schema_raw)
    document = parse_file(document_path)
    truth = json.loads(truth_path.read_text())
    context_doc, note = select_context(document, list(schema.fields.values()), max_chars=6000)
    messages = build_messages(schema, context_doc)
    extra = method_kwargs(method, schema_raw)

    metrics: dict = {
        "experiment": exp_id, "model": model, "category": category, "method": method,
        "document": document_path.name, "calls": 0, "tokens_in": 0, "tokens_out": 0,
        "cost_usd": 0.0, "reserved_usd": 0.0, "latency_ms": 0,
    }
    budget.begin_experiment()
    if not budget.may_call():
        metrics.update(status="skipped_budget", failure="experiment/aggregate cap reached")
        return metrics

    started = time.monotonic()
    failure = None
    contents: list[str] = []
    try:
        response = gateway.complete(model, messages, extra_params=method_kwargs(method, schema_raw))
    except LlmBudgetError as exc:
        metrics.update(status="budget_refused", failure=str(exc)[:200])
        return metrics
    except (LlmError, LlmBadResponseError) as exc:
        metrics.update(status="error", failure=str(exc)[:200])
        return metrics
    metrics["calls"] += 1
    metrics["tokens_in"] += response.tokens_in
    metrics["tokens_out"] += response.tokens_out
    metrics["latency_ms"] += int((time.monotonic() - started) * 1000)
    actual = usage_cost(response.raw)
    metrics["cost_usd"] += actual or 0.0
    content = _extract_content(response.raw, method)
    if content is None:
        metrics.update(status="error", failure="no content (reasoning/truncation)")
        return metrics
    contents.append(content)

    parsed = None
    parse_error = None
    try:
        parsed = parse_extraction_response(content, schema)
    except Exception as exc:  # parse ExtractionParseError and json quirks
        parse_error = str(exc)[:200]
    if parsed is None and repair:
        repair_messages = messages + [
            {"role": "assistant", "content": content[-4000:]},
            {"role": "user", "content":
                "The previous response was not valid JSON (it may have been truncated)."
                " Return ONLY the corrected, complete JSON object now."},
        ]
        started_r = time.monotonic()
        try:
            response = gateway.complete(model, repair_messages, extra_params=method_kwargs(method, schema_raw))
        except (LlmError, LlmBadResponseError) as exc:
            metrics.update(status="error", failure=f"repair failed: {str(exc)[:150]}")
            return metrics
        metrics["calls"] += 1
        metrics["tokens_in"] += response.tokens_in
        metrics["tokens_out"] += response.tokens_out
        metrics["latency_ms"] += int((time.monotonic() - started_r) * 1000)
        actual = usage_cost(response.raw)
        metrics["cost_usd"] += actual or 0.0
        content = _extract_content(response.raw, method)
        if content is not None:
            contents.append(content)
            try:
                parsed = parse_extraction_response(content, schema)
                parse_error = None
            except Exception as exc:
                parse_error = str(exc)[:200]
    if parsed is None:
        metrics.update(status="parse_error", failure=parse_error or "unparseable")
        return metrics

    contract = result_to_contract(
        ExtractionResult(entries=parsed.entries, provenance={k: "llm" for k in schema.fields}),
        schema_raw.get("name", ""), latency_ms=metrics["latency_ms"],
    )
    contract["document"] = document_path.stem
    errors = validate.validate_contract(contract, schema_raw)
    scored = validate.score(contract, truth, schema_raw)
    total_cells = max(len(schema.fields) * max(len(parsed.entries), 1), 1)
    nonnull = sum(
        1 for entry in parsed.entries for fr in entry.values() if fr.value is not None
    )
    cited = sum(
        1 for entry in parsed.entries for fr in entry.values()
        if fr.value is not None and fr.citation
    )
    metrics.update({
        "status": "scored",
        "contract_valid": not errors,
        "contract_errors": errors[:5],
        "validator_score": scored["score"],
        "row_accuracy": round((scored["rows"]["truth"] - len(scored["missing_rows"])) / max(scored["rows"]["truth"], 1), 4),
        "field_accuracy_mean": scored["score"],
        "records": len(parsed.entries),
        "expected_records": scored["rows"]["truth"],
        "field_completeness": round(nonnull / total_cells, 4),
        "missing_field_cells": len(scored["misses"]),
        "hallucinated_cells": len(scored["hallucinations"]),
        "citation_coverage": round(cited / max(nonnull, 1), 4),
        "per_field": scored["per_field"],
        "repair_used": len(contents) > 1,
        "entries_snapshot": contract["entries"],
    })
    return metrics


def run_escalation(
    gateway: LlmGateway,
    budget: Budget,
    exp_id: str,
    cheap: str,
    strong: str,
    method: str,
    document_path: Path,
    schema_path: Path,
    truth_path: Path,
) -> dict:
    """Cheap model first; escalate only the unresolved fields to the strong
    model; merge conservatively (fill nulls, keep first otherwise)."""
    schema_raw = json.loads(schema_path.read_text())
    schema = normalize_schema(schema_raw)
    first = run_variant(gateway, budget, f"{exp_id}-cheap", cheap, "cheap", method,
                        document_path, schema_path, truth_path)
    combined = dict(first)
    combined["experiment"] = exp_id
    combined["escalated_to"] = strong
    if first.get("status") != "scored" or not budget.may_call():
        combined["escalation"] = "skipped"
        return combined
    unresolved = [
        k for k, acc in first.get("per_field", {}).items() if acc < 1.0
    ] or [
        k for k in schema_raw["fields"]
    ]
    # re-extract only unresolved fields with the strong model on the same context
    document = parse_file(document_path)
    schema = normalize_schema(schema_raw)
    truth = json.loads(truth_path.read_text())
    context_doc, _ = select_context(document, [schema.fields[k] for k in unresolved], max_chars=6000)
    sub = type(schema)(name=schema.name, description=schema.description,
                       fields={k: schema.fields[k] for k in unresolved})
    messages = build_messages(sub, context_doc)
    started = time.monotonic()
    try:
        response = gateway.complete(strong, messages, extra_params=method_kwargs(method, schema_raw))
    except (LlmError, LlmBadResponseError) as exc:
        combined["escalation"] = f"failed: {str(exc)[:120]}"
        return combined
    combined["calls"] += 1
    combined["tokens_in"] += response.tokens_in
    combined["tokens_out"] += response.tokens_out
    combined["latency_ms"] += int((time.monotonic() - started) * 1000)
    actual = usage_cost(response.raw)
    combined["cost_usd"] += actual or 0.0
    content = _extract_content(response.raw, method)
    try:
        parsed = parse_extraction_response(content, sub)
    except Exception as exc:
        combined["escalation"] = f"parse failed: {str(exc)[:120]}"
        return combined
    merged = 0
    for entry in first.get("entries_snapshot", []):
        pass
    # merge by primary (fill nulls only)
    pkey = schema.primary_key
    from app.evaluation.serialize import field_result_to_dict

    strong_by_primary = {}
    for entry in parsed.entries:
        serialized = {k: field_result_to_dict(fr) for k, fr in entry.items()}
        pfr = entry.get(pkey)
        pval = pfr.value if pfr else None
        if pval is not None:
            strong_by_primary[str(pval)] = serialized
    escalations = {"filled": 0, "conflicts": 0}
    entries = json.loads(json.dumps(first.get("entries_snapshot") or []))
    if not entries:
        combined["escalation"] = "no baseline entries to merge"
        return combined
    for entry in entries:
        pv = entry.get(pkey, {}).get("value")
        strong_entry = strong_by_primary.get(str(pv))
        if strong_entry is None:
            continue
        for k, fr in entry.items():
            if fr["value"] is None:
                replacement = strong_entry.get(k)
                if replacement is not None and replacement.get("value") is not None:
                    entry[k] = replacement
                    escalations["filled"] += 1
            elif strong_entry is not None and strong_entry.get(k, {}).get("value") not in (None, fr["value"]):
                escalations["conflicts"] += 1
    combined["escalation"] = escalations
    contract = {"schema": schema_raw.get("name", ""), "entries": entries,
                "metadata": {"model_calls": combined["calls"], "tokens_in": combined["tokens_in"],
                             "tokens_out": combined["tokens_out"], "cost_usd": combined["cost_usd"],
                             "latency_ms": combined["latency_ms"]}}
    contract["document"] = document_path.stem
    errors = validate.validate_contract(contract, schema_raw)
    scored = validate.score(contract, truth, schema_raw)
    combined.update({
        "status": "scored",
        "contract_valid": not errors,
        "validator_score": scored["score"],
        "row_accuracy": round((scored["rows"]["truth"] - len(scored["missing_rows"])) / max(scored["rows"]["truth"], 1), 4),
        "records": len(entries),
        "field_completeness": round(
            sum(1 for e in entries for fr in e.values() if fr["value"] is not None)
            / max(len(schema_raw["fields"]) * len(entries), 1), 4),
    })
    return combined


# json_schema response_format: works on openai-provider models (luna), but
# intermittently 400s on anthropic-provider models (9/9 accepted, then
# consistent 400s) -> anthropic models run with json_prompt only.
EXPERIMENTS = [
    ("E1-haiku-json-prompt", BASELINE_MODEL, "baseline", "json_prompt", "budget_001"),
    ("E2-luna-json-schema", CHEAP_MODEL, "cheap", "json_schema", "budget_001"),
    ("E3-sonnet-json-prompt", STRONG_MODEL, "strong", "json_prompt", "budget_001"),
    ("E4-haiku-json-prompt-t12", BASELINE_MODEL, "baseline", "json_prompt", "t12_001"),
    ("E5-luna-json-prompt-t12", CHEAP_MODEL, "cheap", "json_prompt", "t12_001"),
    ("E6-luna-to-sonnet-escalation", CHEAP_MODEL, "routing", "json_prompt", "budget_001"),
]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "run"], default="run")
    args = ap.parse_args(argv)

    ledger = CostLedger(cap_usd=AGGREGATE_CAP_USD)
    gateway = LlmGateway(ledger=ledger)
    budget = Budget(ledger, AGGREGATE_CAP_USD, PER_EXPERIMENT_CAP_USD)

    if args.stage == "probe":
        rows = []
        for model in (BASELINE_MODEL, CHEAP_MODEL, STRONG_MODEL):
            for method in ("json_prompt", "json_schema", "tools"):
                row = probe(gateway, model, method)
                rows.append(row)
                print(f"{model:32} {method:12} accepted={row['accepted']}")
        (HERE / "probes.json").write_text(json.dumps(rows, indent=2))
        return 0

    all_metrics = []
    for exp_id, model, category, method, document in EXPERIMENTS:
        doc_path = SAMPLES / f"{document}.pdf"
        if not doc_path.exists():
            doc_path = SAMPLES / f"{document}.{document.split('_')[0]}"
        doc_path = next(SAMPLES.glob(f"{document}.*"))
        # schema name comes from the manifest mapping (budget_001 -> budget_variance, t12_001 -> operating_statement_lines)
        schema_name = {"budget_001": "budget_variance", "t12_001": "operating_statement_lines"}[document]
        schema_path = SCHEMAS / f"{schema_name}.json"
        truth_path = TRUTH / f"{document}.json"
        budget.begin_experiment()
        if exp_id.endswith("escalation"):
            metrics = run_escalation(gateway, budget, exp_id, model, STRONG_MODEL,
                                     method, doc_path, schema_path, truth_path)
        else:
            metrics = run_variant(gateway, budget, exp_id, model, category, method,
                                  doc_path, schema_path, truth_path)
        all_metrics.append(metrics)
        (HERE / f"{exp_id}.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(f"{exp_id:32} {metrics.get('status'):14} score={metrics.get('validator_score','-')}"
              f" records={metrics.get('records','-')}/{metrics.get('expected_records','-')}"
              f" cost=${metrics.get('cost_usd', 0):.4f} calls={metrics.get('calls')}")
        if ledger.exposure_usd >= AGGREGATE_CAP_USD:
            print("AGGREGATE CAP REACHED — stopping")
            break

    (HERE / "RESULTS.json").write_text(json.dumps(all_metrics, indent=2, default=str))
    print(f"\naggregate exposure: ${ledger.exposure_usd:.4f} / ${AGGREGATE_CAP_USD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
