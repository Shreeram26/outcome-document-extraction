"""Typed value parsing shared by deterministic extraction.

Converts raw cell/line values into contract-shaped values: JSON numbers
(currency and thousands separators stripped), YYYY-MM-DD dates, booleans, or
trimmed strings. Values that look like placeholders (N/A) yield None."""
from __future__ import annotations

import re
from datetime import date, datetime

DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y",
    "%d %b %Y", "%d %B %Y", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S", "%m-%d-%Y",
]
_MONTH_FIXES = {"sept": "sep"}
_NULL_WORDS = {"n/a", "na", "-", "--", "none", "null", "tbd", ""}
_TRUE_WORDS = {"true", "yes", "y"}
_FALSE_WORDS = {"false", "no", "n"}
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_CURRENCY_CHARS = " $€£₹"


def parse_scalar(raw) -> tuple[Any, str] | None:
    """Parse a raw value into (typed_value, kind) with kind in
    {"number", "date", "bool", "string"}; None when there is no value."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw, "bool"
    if isinstance(raw, datetime):
        return raw.date().isoformat(), "date"
    if isinstance(raw, date):
        return raw.isoformat(), "date"
    if isinstance(raw, (int, float)):
        return _as_number(float(raw)), "number"
    text = " ".join(str(raw).split())
    if text.lower() in _NULL_WORDS:
        return None
    low = text.lower()
    if low in _TRUE_WORDS:
        return True, "bool"
    if low in _FALSE_WORDS:
        return False, "bool"
    iso = _try_date(text)
    if iso:
        return iso, "date"
    number = _try_number(text)
    if number is not None:
        return number, "number"
    return text, "string"


def _try_date(text: str) -> str | None:
    parts = text.split()
    if parts and parts[0].lower() in _MONTH_FIXES:
        parts[0] = _MONTH_FIXES[parts[0].lower()]
        text = " ".join(parts)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _try_number(text: str) -> int | float | None:
    t = text
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1]
    if t.endswith("%"):
        t = t[:-1]
    t = t.replace(",", "").strip(_CURRENCY_CHARS)
    if not _NUMBER_RE.match(t):
        return None
    x = float(t)
    if neg:
        x = -x
    return _as_number(x)


def _as_number(x: float) -> int | float:
    return int(x) if float(x).is_integer() else float(x)
