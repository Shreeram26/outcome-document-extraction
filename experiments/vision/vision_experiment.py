"""Controlled vision experiment: compare 3 allowed models on scanned pages
from lease_001. Uses the existing gateway, CostLedger, and budget safeguards.

Usage: python experiments/vision/vision_experiment.py
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.extraction import (  # noqa: E402
    LlmBadResponseError,
    LlmBudgetError,
    LlmError,
    LlmGateway,
    normalize_schema,
    parse_extraction_response,
)
from app.extraction.ledger import CostLedger, usage_cost
from app.extraction.llm import ALLOWED_MODELS  # noqa: E402
from tools import validate  # noqa: E402

AGGREGATE_CAP_USD = 0.15
PER_MODEL_CAP_USD = 0.06
SAMPLES = REPO / "sample" / "sample-data-documents"
SCHEMAS = REPO / "sample" / "schemas"
TRUTH = REPO / "sample" / "sample-data-output-validated"
HERE = Path(__file__).resolve().parent

DOCUMENT = "lease_001"
SELECTED_PAGES = [2, 18]          # 1-based; 20 + 7 cited fields
RENDER_DPI = 200
MODELS = [
    ("anthropic/claude-haiku-5.5", "baseline"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "vl-specialist"),
    ("google/gemini-3.8-flash", "flash"),
]
SCHEMA_PATH = SCHEMAS / "lease_abstraction.json"
TRUTH_PATH = TRUTH / f"{DOCUMENT}.json"


def _load_env():
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")


def render_pages(pdf_path: Path, pages: list[int], dpi: int = RENDER_DPI) -> dict[int, str]:
    """Render selected pages to base64-encoded PNG. Returns {page_number: b64}."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    result = {}
    for pn in pages:
        page = doc[pn - 1]
        pix = page.get_pixmap(dpi=dpi)
        result[pn] = base64.b64encode(pix.tobytes("png")).decode("ascii")
    doc.close()
    return result


def build_vision_messages(
    schema_raw: dict, page_images: dict[int, str], page_numbers: list[int]
) -> list[dict]:
    """Build multimodal messages: text prompt + page images.

    Uses the OpenAI-compatible content-array format that OpenRouter
    normalises for all vision-capable models."""
    schema = schema_raw["fields"]
    field_lines = []
    for key, f in schema.items():
        field_lines.append(f'- "{key}" ({f["name"]}): {f["description"]}')
    page_list = ", ".join(str(p) for p in page_numbers)
    system = (
        "You are a precise document extraction engine. You will see scanned"
        f" pages {page_list} of a commercial lease document. Extract the"
        " requested fields from the visible text. Rules:"
        "\n1. Use only what is visible in the images. Never guess."
        "\n2. If a value is not visible, set \"value\" to null, \"citation\""
        " to \"\" and \"confidence\" to 0."
        "\n3. Numbers are JSON numbers (no currency symbols, no commas)."
        "\n4. Dates are YYYY-MM-DD strings."
        "\n5. \"citation\" is a short verbatim snippet prefixed with its page"
        " location, e.g. \"[Page 2] Base Rent: $4,250.00\". At most 60"
        " characters."
        "\n6. \"confidence\" is 0.0-1.0."
        "\n7. Reply with ONLY a JSON object: {\"entries\": [{...one entry with"
        " every field...}]}"
    )
    user_text = (
        f"Extract the following fields from these scanned lease pages"
        f" (pages {page_list}):\n\n" + "\n".join(field_lines)
    )
    content: list[dict] = [{"type": "text", "text": user_text}]
    for pn in page_numbers:
        b64 = page_images[pn]
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def estimate_vision_cost(
    messages: list[dict], model: str, pricing: dict, num_images: int,
    max_tokens: int = 8000, safety_factor: float = 1.5,
) -> float:
    """Conservative cost estimate for a multimodal call.
    Image tokens: ~750 tokens per 512×512 tile; a 200-dpi page ≈ 2-4 tiles.
    Conservative: assume 3000 tokens per image."""
    price = pricing.get(model, pricing.get("default", {"input": 3.0, "output": 15.0}))
    text_tokens = 0
    image_tokens = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            text_tokens += math.ceil(len(content) / 4)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_tokens += math.ceil(len(block.get("text", "")) / 4)
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    image_tokens += 3000  # conservative per-image estimate
    in_tokens = text_tokens + image_tokens + 32
    out_tokens = max_tokens
    est = (in_tokens * price["input"] + out_tokens * price["output"]) / 1_000_000
    return est * safety_factor


def score_against_truth(
    parsed_entries: list[dict], truth_path: Path, schema_raw: dict
) -> dict:
    """Score extracted entries against ground truth using the assignment validator."""
    truth = json.loads(truth_path.read_text())
    t_entry = truth["entries"][0]
    fields = list(schema_raw["fields"].keys())
    correct = miss = halluc = 0
    nonnull = cited = 0
    for key in fields:
        tv = t_entry.get(key, {}).get("value")
        ov = None
        for entry in parsed_entries:
            fr = entry.get(key)
            if fr is not None and fr.value is not None:
                ov = fr.value
                break
        if tv is None and ov is None:
            correct += 1
        elif tv is not None and ov is None:
            miss += 1
        elif tv is None and ov is not None:
            halluc += 1
        else:
            try:
                if abs(float(tv) - float(ov)) <= max(0.01, abs(float(tv)) * 0.005):
                    correct += 1
                else:
                    miss += 1
            except (TypeError, ValueError):
                if str(tv).strip().lower() == str(ov).strip().lower():
                    correct += 1
                else:
                    miss += 1
        if ov is not None:
            nonnull += 1
            fr = next((entry.get(key) for entry in parsed_entries if entry.get(key) and entry[key].value is not None), None)
            if fr and fr.citation:
                cited += 1
    total = len(fields)
    return {
        "fields_total": total,
        "correct": correct,
        "missed": miss,
        "hallucinated": halluc,
        "field_accuracy": round(correct / total, 4) if total else 0,
        "field_completeness": round(nonnull / total, 4) if total else 0,
        "citation_coverage": round(cited / max(nonnull, 1), 4) if nonnull else 0,
    }


def main():
    _load_env()
    schema_raw = json.loads(SCHEMA_PATH.read_text())
    schema = normalize_schema(schema_raw)
    truth_path = TRUTH_PATH
    pdf_path = SAMPLES / f"{DOCUMENT}.pdf"

    print(f"=== Vision experiment: {DOCUMENT} pages {SELECTED_PAGES} at {RENDER_DPI} dpi ===\n")

    # 1. Render pages
    page_images = render_pages(pdf_path, SELECTED_PAGES)
    for pn, b64 in page_images.items():
        print(f"  page {pn}: {len(b64) // 1024}KB base64 ({len(b64) * 3 // 4 // 1024}KB raw)")

    # 2. Build messages (shared across all models)
    messages = build_vision_messages(schema_raw, page_images, SELECTED_PAGES)
    num_images = sum(
        1 for m in messages for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "image_url"
    )
    print(f"  multimodal messages: {num_images} images\n")

    # 3. Ledger + budget
    ledger = CostLedger(cap_usd=AGGREGATE_CAP_USD)
    gateway = LlmGateway(ledger=ledger)
    model_spend = {m: 0.0 for m, _ in MODELS}

    all_results = []
    for model, category in MODELS:
        if model not in ALLOWED_MODELS:
            print(f"SKIP {model}: not in ALLOWED_MODELS")
            continue
        # per-model cap check
        spend_before = ledger.exposure_usd
        if spend_before >= AGGREGATE_CAP_USD:
            print(f"STOP: aggregate cap ${AGGREGATE_CAP_USD} reached")
            break

        est = estimate_vision_cost(messages, model, gateway._pricing, num_images)
        print(f"\n=== {model} ({category}) ===")
        print(f"  estimated cost: ${est:.4f} (exposure before: ${ledger.exposure_usd:.4f})")
        if ledger.exposure_usd + est > AGGREGATE_CAP_USD:
            print(f"  SKIP: would exceed aggregate cap")
            continue

        started = time.monotonic()
        try:
            response = gateway.complete(model, messages, max_tokens=8000)
        except LlmBudgetError as exc:
            print(f"  BUDGET REFUSED: {str(exc)[:120]}")
            all_results.append({"model": model, "status": "budget_refused"})
            continue
        except (LlmTimeoutError, LlmBadResponseError, LlmError) as exc:
            duration = int((time.monotonic() - started) * 1000)
            all_results.append({"model": model, "status": "error", "error": str(exc)[:200], "latency_ms": duration})
            print(f"  ERROR: {str(exc)[:150]}")
            continue

        duration = int((time.monotonic() - started) * 1000)
        actual = usage_cost(response.raw)
        print(f"  SUCCESS: {duration}ms, tokens={response.tokens_in}/{response.tokens_out}, cost=${actual or 0:.4f}")

        # Parse and score
        parsed = None
        parse_error = None
        try:
            parsed = parse_extraction_response(response.content, schema)
        except Exception as exc:
            parse_error = str(exc)[:200]

        if parsed is None:
            all_results.append({
                "model": model, "category": category, "status": "parse_error",
                "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
                "cost_usd": actual or 0, "latency_ms": duration,
                "parse_error": parse_error,
            })
            print(f"  PARSE ERROR: {parse_error}")
            continue

        entries = parsed.entries
        scores = score_against_truth(entries, truth_path, schema_raw)
        page_refs = set()
        for entry in entries:
            for fr in entry.values():
                for pn in fr.page_numbers or []:
                    page_refs.add(pn)

        result = {
            "model": model, "category": category, "status": "scored",
            "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
            "cost_usd": actual or 0, "latency_ms": duration,
            "finish_reason": getattr(response.raw.choices[0], "finish_reason", None) if response.raw else None,
            **scores,
            "page_references": sorted(page_refs),
            "sample_values": {
                k: entry_fields[0].value if (entry_fields := [entry[k]]) else None
                for k in list(schema.fields.keys())[:5]
            },
        }
        all_results.append(result)
        print(f"  field_accuracy: {scores['field_accuracy']}"
              f" | completeness: {scores['field_completeness']}"
              f" | missed: {scores['missed']} | halluc: {scores['hallucinated']}"
              f" | citations: {scores['citation_coverage']}")

    # 4. Summary
    print(f"\n=== SUMMARY ===")
    print(f"aggregate exposure: ${ledger.exposure_usd:.4f} / ${AGGREGATE_CAP_USD}")
    print(f"settled: ${ledger.settled_usd:.4f}")
    for r in all_results:
        model = r.get("model", "?")
        status = r.get("status", "?")
        acc = r.get("field_accuracy", "-")
        comp = r.get("field_completeness", "-")
        cost = r.get("cost_usd", "-")
        print(f"  {model:40} {status:12} acc={acc} compl={comp} cost=${cost}")
    (HERE / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (HERE / "ledger-status.json").write_text(json.dumps(ledger.status(), indent=2))
    print(f"\nresults saved to {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    _load_env()
    raise SystemExit(main())
