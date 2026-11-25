from __future__ import annotations

from datetime import datetime

from app.extraction import parse_scalar


def test_numbers():
    assert parse_scalar("1,425") == (1425, "number")
    assert parse_scalar("$4,250.00") == (4250, "number")
    assert parse_scalar("€900.50") == (900.5, "number")
    assert parse_scalar("(500)") == (-500, "number")
    assert parse_scalar("91.67%") == (91.67, "number")
    assert parse_scalar(42) == (42, "number")
    assert parse_scalar(1425.5) == (1425.5, "number")


def test_dates():
    assert parse_scalar("2026-09-27") == ("2026-09-27", "date")
    assert parse_scalar("09/27/2026") == ("2026-09-27", "date")
    assert parse_scalar("March 4, 2010") == ("2010-03-04", "date")
    assert parse_scalar("4 March 2010") == ("2010-03-04", "date")
    assert parse_scalar("Sept 3, 2026") == ("2026-09-03", "date")
    assert parse_scalar(datetime(2026, 9, 27, 12, 0)) == ("2026-09-27", "date")


def test_booleans():
    assert parse_scalar("Yes") == (True, "bool")
    assert parse_scalar("true") == (True, "bool")
    assert parse_scalar("No") == (False, "bool")
    assert parse_scalar(True) == (True, "bool")


def test_strings_are_left_alone():
    assert parse_scalar("510-500") == ("510-500", "string")
    assert parse_scalar("417 .417") == ("417 .417", "string")
    assert parse_scalar("Suite 214W") == ("Suite 214W", "string")
    assert parse_scalar("  P1 -1x1 Pearl  ") == ("P1 -1x1 Pearl", "string")


def test_null_words_and_empty():
    for raw in ("", "   ", "N/A", "na", "-", "--", "None", "null"):
        assert parse_scalar(raw) is None
    assert parse_scalar(None) is None
