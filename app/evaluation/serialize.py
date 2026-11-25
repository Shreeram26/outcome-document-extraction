"""Serialization of ExtractionResult into the assignment's output contract shape
({schema, entries, metadata}), ready for tools/validate.py."""
from __future__ import annotations

from app.extraction.engine import ExtractionResult
from app.extraction.llm import usage_cost


def field_result_to_dict(fr) -> dict:
    confidence = max(0.0, min(1.0, float(fr.confidence)))
    value = fr.value
    if value is None:
        confidence = 0.0
    d: dict = {
        "value": value,
        "citation": fr.citation if value is not None else "",
        "confidence": round(confidence, 4),
    }
    if value is not None:
        if fr.page_numbers:
            d["page_numbers"] = list(fr.page_numbers)
        if fr.source_locations:
            d["source_locations"] = list(fr.source_locations)
    return d


def result_to_contract(result: ExtractionResult, schema_name: str, latency_ms: int) -> dict:
    """Contract-shaped result: top-level schema key, entries carrying every field,
    and metadata with model calls, tokens, cost and latency.

    Cost reporting is conservative: `cost_usd` is the total EXPOSURE for the
    document (settled provider-reported cost plus any retained reservations,
    e.g. timed-out calls that may still be billed), while `cost_settled_usd` is
    only the provider-confirmed portion."""
    entries = [
        {key: field_result_to_dict(fr) for key, fr in entry.items()} for entry in result.entries
    ]
    models: dict[str, dict] = {}
    for response in result.llm_responses:
        entry = models.setdefault(
            response.model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "settled_cost_usd": 0.0}
        )
        entry["calls"] += 1
        entry["tokens_in"] += response.tokens_in
        entry["tokens_out"] += response.tokens_out
        actual = usage_cost(response)
        if actual is not None:
            entry["settled_cost_usd"] += actual
    metadata: dict = {
        "model_calls": len(result.call_log),
        "tokens_in": sum(m["tokens_in"] for m in models.values()),
        "tokens_out": sum(m["tokens_out"] for m in models.values()),
        "cost_usd": round(result.cost_exposure_usd, 6),
        "cost_settled_usd": round(result.cost_settled_usd, 6),
        "latency_ms": int(latency_ms),
        "calls": list(result.call_log),
    }
    if result.conflicts:
        metadata["conflicts"] = list(result.conflicts)
    if models:
        metadata["models"] = {
            name: {**m, "settled_cost_usd": round(m["settled_cost_usd"], 6)}
            for name, m in models.items()
        }
    return {"schema": schema_name, "entries": entries, "metadata": metadata}
