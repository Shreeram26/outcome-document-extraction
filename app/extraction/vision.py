"""Generic scanned-page vision support: rendering + multimodal message building.

When a document's pages have no extractable text layer (scanned PDFs), the
pages are rendered to PNG images and sent to a vision-capable LLM through the
existing gateway. Page numbers from the original document are preserved in
citations, so extracted values carry the same `[Page N]` reference format as
text-based extraction."""
from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any

import pymupdf

from ..parsing.models import ParsedDocument

DEFAULT_VISION_DPI = 200
DEFAULT_VISION_BATCH_SIZE = 5


def render_scanned_pages(
    document: ParsedDocument,
    dpi: int = DEFAULT_VISION_DPI,
    max_pages: int = 0,
) -> dict[int, str]:
    """Render scanned pages to base64-encoded PNG images.

    Args:
        document: ParsedDocument with `scanned_pages` and `source_path`.
        dpi: Rendering resolution (higher = more tokens = higher cost).
        max_pages: Maximum pages to render (0 = all scanned pages).

    Returns:
        {page_number: base64_png} for each successfully rendered scanned page.
    """
    source = getattr(document, "source_path", None)
    if not source:
        return {}
    doc = pymupdf.open(source)
    result: dict[int, str] = {}
    rendered = 0
    for pn in document.scanned_pages:
        if max_pages and rendered >= max_pages:
            break
        if pn > len(doc):
            continue
        page = doc[pn - 1]
        pix = page.get_pixmap(dpi=dpi)
        result[pn] = base64.b64encode(pix.tobytes("png")).decode("ascii")
        rendered += 1
    doc.close()
    return result


def build_vision_messages(
    schema_fields: dict[str, dict[str, Any]],
    page_images: dict[int, str],
    page_numbers: list[int],
) -> list[dict]:
    """Build multimodal messages for vision-based extraction.

    Args:
        schema_fields: The schema field definitions (name/description map).
        page_images: {page_number: base64_png} from `render_scanned_pages`.
        page_numbers: The original page numbers, in order.

    Returns:
        OpenAI-compatible multimodal messages for the gateway.
    """
    field_lines = []
    for key, f in schema_fields.items():
        desc = f.get("description", f.get("name", ""))
        field_lines.append(f'- "{key}" ({f.get("name", key)}): {desc}')
    page_list = ", ".join(f"page {pn}" for pn in page_numbers)
    system = (
        "You are a precise document extraction engine. You will see scanned"
        f" pages of a document ({page_list}). Extract the requested fields"
        " from the visible text. Rules:"
        "\n1. Use only what is visible in the images. Never guess."
        "\n2. If a value is not visible, set \"value\" to null, \"citation\""
        " to \"\" and \"confidence\" to 0."
        "\n3. Numbers are JSON numbers (no currency symbols, no commas)."
        "\n4. Dates are YYYY-MM-DD strings."
        "\n5. \"citation\" is a short verbatim snippet prefixed with its page"
        " location, e.g. \"[Page 2] Base Rent: $4,250.00\". At most 60"
        " characters."
        "\n6. \"confidence\" is 0.0-1.0."
        "\n7. Reply with ONLY a JSON object:"
        " {\"entries\": [{...one entry with every field...}]}"
        "\n8. If the document contains multiple records (table rows, list"
        " items), return one entry per record. Otherwise return exactly one"
        " entry."
    )
    user_text = (
        "Extract the following fields from these scanned pages:\n\n"
        + "\n".join(field_lines)
    )
    content: list[dict] = [{"type": "text", "text": user_text}]
    for pn in page_numbers:
        b64 = page_images.get(pn)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def estimate_vision_cost(
    messages: list[dict], model: str, pricing: dict[str, dict[str, float]],
    num_images: int, max_tokens: int = 8000, safety_factor: float = 1.5,
) -> float:
    """Conservative cost estimate for a multimodal call. Image tokens are
    estimated at 3000 per image (covers most 200-dpi pages after model-side
    resizing)."""
    price = pricing.get(model, pricing.get("default", {"input": 3.0, "output": 15.0}))
    text_tokens = 32
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            text_tokens += math.ceil(len(content) / 4)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_tokens += math.ceil(len(block.get("text", "")) / 4)
    image_tokens = num_images * 3000
    in_tokens = text_tokens + image_tokens
    out_tokens = max_tokens
    est = (in_tokens * price["input"] + out_tokens * price["output"]) / 1_000_000
    return est * safety_factor


def vision_page_batches(
    scanned_pages: list[int], batch_size: int = DEFAULT_VISION_BATCH_SIZE
) -> list[list[int]]:
    """Split scanned page numbers into batches for sequential vision calls."""
    if batch_size <= 0:
        batch_size = DEFAULT_VISION_BATCH_SIZE
    return [scanned_pages[i:i + batch_size] for i in range(0, len(scanned_pages), batch_size)]
