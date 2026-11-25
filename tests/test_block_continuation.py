"""Tests for generic block-continuation table extraction.

A primary record row followed by rows with an empty primary cell (charge
lines, detail lines) must have those rows' values attached to the record —
driven structurally and by schema-description phrases, never by hardcoded
labels. All names/values here are synthetic and generic."""
from __future__ import annotations

import json

import pytest

from app.extraction import normalize_schema
from app.extraction.deterministic import deterministic_extract
from app.parsing import parse_file
from app.parsing.models import Cell, ParsedDocument, Sheet


def _ledger_schema():
    """Generic ledger schema: a primary identifier column, a category column
    referenced by quoted phrases, an amount column, and derived fields."""
    return normalize_schema(
        {
            "name": "ledger",
            "fields": {
                "record_id": {
                    "name": "Record Id", "description": "The 'Record Id' column.", "primary": True,
                },
                "base_amount": {
                    "name": "Base Amount",
                    "description": "The amount on the record's 'BASE' line, as a number.",
                },
                "extra_fees": {
                    "name": "Extra Fees",
                    "description": "Sum of all other fee lines for the record other than 'BASE', as a number. 0 if none.",
                },
                "grand_total": {
                    "name": "Grand Total",
                    "description": "The amount on the record's 'GRAND TOTAL' line, as a number.",
                },
                "free_note": {"name": "Free Note", "description": "A note field with no default semantics."},
            },
        }
    )


def _ledger_sheet(rows):
    return Sheet(name="L", rows=tuple(rows), n_rows=len(rows), n_cols=max((len(r) for r in rows), default=0))


def _header():
    return (Cell("Record Id", "A1"), Cell("Line Type", "B1"), Cell("Amount", "C1"))


def _row(r, record_id, line_type, amount):
    return (
        Cell(record_id, f"A{r}"),
        Cell(line_type, f"B{r}"),
        Cell(amount, f"C{r}"),
    )


def test_continuation_row_with_quoted_category_fills_field():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "BASE", None),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    entry = result.entries[0]
    assert entry["base_amount"].value == 100


def test_primary_row_values_extracted_normally():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert result.entries[0]["record_id"].value == "R-1"
    assert result.entries[0]["base_amount"].value == 100


def test_category_row_value_maps_to_quoted_field():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "GRAND TOTAL", 250),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert result.entries[0]["grand_total"].value == 250
    assert result.entries[0]["grand_total"].citation == "[Sheet: L, Cell: C3] 250"
    assert result.entries[0]["grand_total"].source_locations == ["L!C3"]


def test_multiple_continuation_rows_for_one_record():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "FEE-A", 5),
        _row(4, None, "FEE-B", 7),
        _row(5, None, "GRAND TOTAL", 12),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert len(result.entries) == 1
    assert result.entries[0]["grand_total"].value == 12


def test_unrelated_category_row_does_not_map():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "SOMETHING ELSE", 42),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert result.entries[0]["grand_total"].value is None


def test_aggregate_field_sums_unclaimed_rows_excluding_quoted_category():
    schema = normalize_schema(
        {
            "name": "ledger",
            "fields": {
                "record_id": {
                    "name": "Record Id", "description": "The 'Record Id' column.", "primary": True,
                },
                "base_amount": {
                    "name": "Base Amount",
                    "description": "The amount on the record's 'BASE' line, as a number.",
                },
                "other_fees": {
                    "name": "Other Fees",
                    "description": "Sum of all fee lines for the record other than 'BASE', as a number. 0 if none.",
                },
            },
        }
    )
    sheet = _ledger_sheet([
        (Cell("Record Id", "A1"), Cell("Base", "B1"), Cell("Amount", "C1")),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "FEE-A", 5),
        _row(4, None, "FEE-B", 7),
        _row(5, None, "FEE-C", 2),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(schema, doc)
    assert result.mode == "table"
    assert result.entries[0]["other_fees"].value == 14
    assert result.entries[0]["other_fees"].source_locations == ["L!C3", "L!C4", "L!C5"]


def test_no_aggregation_without_sum_phrasing():
    schema = normalize_schema(
        {
            "name": "ledger",
            "fields": {
                "record_id": {
                    "name": "Record Id", "description": "The 'Record Id' column.", "primary": True,
                },
                "base_amount": {
                    "name": "Base Amount",
                    "description": "The amount on the record's 'BASE' line, as a number.",
                },
                "other_fees": {
                    "name": "Other Fees",
                    "description": "Fee lines for the record. Empty list if none.",
                },
            },
        }
    )
    sheet = _ledger_sheet([
        (Cell("Record Id", "A1"), Cell("Base", "B1"), Cell("Amount", "C1")),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "FEE-A", 5),
        _row(4, None, "FEE-B", 7),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(schema, doc)
    assert result.entries[0]["other_fees"].value is None


def test_ambiguous_category_cell_maps_to_nothing():
    schema = normalize_schema(
        {
            "name": "ledger",
            "fields": {
                "record_id": {
                    "name": "Record Id", "description": "The 'Record Id' column.", "primary": True,
                },
                "base_amount": {
                    "name": "Base Amount",
                    "description": "The amount on the record's 'BASE' line, as a number.",
                },
                "alt_amount": {
                    "name": "Alt Amount",
                    "description": "The amount on the record's 'BASE ALT' line, as a number.",
                },
                "line_type": {"name": "Line Type", "description": "The 'Line Type' column."},
            },
        }
    )
    sheet = _ledger_sheet([
        (Cell("Record Id", "A1"), Cell("Line Type", "B1"), Cell("Amount", "C1")),
        _row(2, "R-1", "BASE", 100),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(schema, doc)
    entry = result.entries[0]
    assert entry["alt_amount"].value is None


def test_flat_table_without_continuations_unchanged():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", 10, 11),
        _row(3, "R-2", 20, 21),
        _row(4, "R-3", 30, 31),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    schema = normalize_schema(
        {
            "name": "flat",
            "fields": {
                "record_id": {"name": "Record Id", "description": "The 'Record Id' column.", "primary": True},
                "value_a": {"name": "Line Type", "description": "The 'Line Type' column."},
                "value_b": {"name": "Amount", "description": "The 'Amount' column."},
            },
        }
    )
    result = deterministic_extract(schema, doc)
    assert result.mode == "table"
    assert len(result.entries) == 3
    assert [e["record_id"].value for e in result.entries] == ["R-1", "R-2", "R-3"]
    assert [e["value_a"].value for e in result.entries] == [10, 20, 30]
    assert [e["value_b"].value for e in result.entries] == [11, 21, 31]


def test_multiple_primary_records_with_own_continuation_blocks():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "GRAND TOTAL", 150),
        _row(4, "R-2", "BASE", 200),
        _row(5, None, "GRAND TOTAL", 260),
        _row(6, "R-3", "BASE", 300),
        _row(7, None, "GRAND TOTAL", 310),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert len(result.entries) == 3
    assert [e["record_id"].value for e in result.entries] == ["R-1", "R-2", "R-3"]
    assert [e["grand_total"].value for e in result.entries] == [150, 260, 310]
    assert [e["base_amount"].value for e in result.entries] == [100, 200, 300]


def test_continuation_rows_do_not_leak_across_sheets():
    sheet_a = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "GRAND TOTAL", 150),
    ])
    sheet_b = _ledger_sheet([
        _header(),
        _row(2, "R-9", "GRAND TOTAL", 999),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet_a, sheet_b])
    result = deterministic_extract(_ledger_schema(), doc)
    assert len(result.entries) >= 1
    r = result.entries[0]
    assert r["record_id"].value in ("R-1", "R-9")
    assert r["grand_total"].value != 999 or r["record_id"].value == "R-9"


def test_summary_rows_terminate_continuation_blocks():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, "Total", "TOTAL", 500),
        _row(4, None, "GRAND TOTAL", 999),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    schema = normalize_schema(
        {
            "name": "ledger",
            "fields": {
                "record_id": {"name": "Record Id", "description": "The 'Record Id' column.", "primary": True},
                "base_amount": {"name": "Base Amount", "description": "The 'Base Amount' column."},
                "grand_total": {"name": "Grand Total", "description": "The 'Grand Total' column."},
            },
        }
    )
    result = deterministic_extract(schema, doc)
    assert len(result.entries) == 1
    assert result.entries[0]["grand_total"].value is None


def test_duplicate_continuation_values_do_not_overwrite():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "GRAND TOTAL", 250),
        _row(4, None, "GRAND TOTAL", 999),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    assert len(result.entries) == 1
    assert result.entries[0]["grand_total"].value == 250


def test_continuation_citations_preserve_source_locations():
    sheet = _ledger_sheet([
        _header(),
        _row(2, "R-1", "BASE", 100),
        _row(3, None, "GRAND TOTAL", 250),
    ])
    doc = ParsedDocument(kind="xlsx", filename="l.xlsx", sheets=[sheet])
    result = deterministic_extract(_ledger_schema(), doc)
    fr = result.entries[0]["grand_total"]
    assert fr.citation.startswith("[Sheet: L, Cell: C3]")
    assert fr.source_locations == ["L!C3"]


def test_generic_ledger_regression_block_structured_charges():
    """Regression for the confirmed structural failure: a block-structured
    table where each record's amounts live on continuation rows under a
    category column. Synthetic names throughout."""
    schema = normalize_schema(
        {
            "name": "charges",
            "fields": {
                "account": {"name": "Account", "description": "The 'Account' column.", "primary": True},
                "primary_amount": {"name": "Primary Amount", "description": "The 'Primary Amount' column."},
                "service_amount": {
                    "name": "Service Amount",
                    "description": "The amount on the record's 'SERVICE' line, as a number.",
                },
                "tax_amount": {
                    "name": "Tax Amount",
                    "description": "The amount on the record's 'TAX' line, as a number.",
                },
                "total_amount": {
                    "name": "Total Amount",
                    "description": "The amount on the record's 'TOTAL' line, as a number.",
                },
                "surcharge_total": {
                    "name": "Surcharge Total",
                    "description": "Sum of all lines for the record other than 'SERVICE', 'TAX' and 'TOTAL', as a number. 0 if none.",
                },
            },
        }
    )
    rows = [
        (Cell("Account", "A1"), Cell("Category", "B1"), Cell("Amount", "C1"), Cell("Primary Amount", "D1")),
        (Cell("ACC-1", "A2"), Cell(None, "B2"), Cell(None, "C2"), Cell(None, "D2")),
        (Cell(None, "A3"), Cell("SERVICE", "B3"), Cell(100, "C3"), Cell(None, "D3")),
        (Cell(None, "A4"), Cell("TAX", "B4"), Cell(10, "C4"), Cell(None, "D4")),
        (Cell(None, "A5"), Cell("SURCHARGE-X", "B5"), Cell(3, "C5"), Cell(None, "D5")),
        (Cell(None, "A6"), Cell("SURCHARGE-Y", "B6"), Cell(4, "C6"), Cell(None, "D6")),
        (Cell(None, "A7"), Cell("TOTAL", "B7"), Cell(117, "C7"), Cell(None, "D7")),
        (Cell("ACC-2", "A8"), Cell(None, "B8"), Cell(None, "C8"), Cell(None, "D8")),
        (Cell(None, "A9"), Cell("SERVICE", "B9"), Cell(200, "C9"), Cell(None, "D9")),
        (Cell(None, "A10"), Cell("TAX", "B10"), Cell(20, "C10"), Cell(None, "D10")),
        (Cell(None, "A11"), Cell("TOTAL", "B11"), Cell(220, "C11"), Cell(None, "D11")),
    ]
    sheet = _ledger_sheet(rows)
    doc = ParsedDocument(kind="xlsx", filename="c.xlsx", sheets=[sheet])
    result = deterministic_extract(schema, doc)
    assert result.mode == "table"
    assert len(result.entries) == 2
    first = result.entries[0]
    assert first["account"].value == "ACC-1"
    assert first["service_amount"].value == 100
    assert first["tax_amount"].value == 10
    assert first["total_amount"].value == 117
    assert first["surcharge_total"].value == 7
    assert first["service_amount"].source_locations == ["L!C3"]
    second = result.entries[1]
    assert second["service_amount"].value == 200
    assert second["total_amount"].value == 220
    assert second["surcharge_total"].value is None  # engine applies the 0 default
