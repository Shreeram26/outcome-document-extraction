from __future__ import annotations

import openpyxl
import pytest

from app.extraction import deterministic_extract, normalize_schema
from pathlib import Path

from app.parsing import parse_file
from app.parsing.models import Cell, Page, ParsedDocument, Sheet


def _pdf_doc(lines_by_page):
    doc = ParsedDocument(kind="pdf", filename="x.pdf")
    doc.pages = [Page(number=i + 1, text="\n".join(lines)) for i, lines in enumerate(lines_by_page)]
    return doc


def test_labelled_pdf_values_resolved_with_citations():
    schema = normalize_schema(
        {
            "fields": {
                "property_name": {"name": "Property Name", "description": "Name of the property."},
                "report_date": {"name": "Report Date", "description": "The report date."},
            }
        }
    )
    doc = _pdf_doc([["Property Name: Cedar Court", "Report Date: 09/27/2026"]])
    det = deterministic_extract(schema, doc)
    assert det.mode == "single"
    entry = det.entries[0]
    assert entry["property_name"].value == "Cedar Court"
    assert entry["property_name"].citation == "[Page 1] Property Name: Cedar Court"
    assert entry["property_name"].page_numbers == [1]
    assert entry["property_name"].confidence == 0.9
    assert entry["report_date"].value == "2026-09-27"
    assert det.doc_level["report_date"].page_numbers == [1]


def test_ambiguous_labels_remain_unresolved():
    schema = normalize_schema({"fields": {"alpha": {"name": "Alpha", "description": ""}}})
    doc = _pdf_doc([["Alpha: first"], ["Alpha: second"]])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["alpha"].value is None


def test_range_like_and_mixed_values_left_unresolved():
    schema = normalize_schema(
        {
            "fields": {
                "date": {"name": "Date", "description": "The date."},
                "rent": {"name": "Rent", "description": "The rent."},
            }
        }
    )
    doc = _pdf_doc([["Date: 09/21/2026-09/27/2026", "Rent: 4,250.00 per month"]])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["date"].value is None
    assert det.entries[0]["rent"].value is None


def test_sheet_label_with_adjacent_value():
    schema = normalize_schema(
        {"fields": {"property_rsf": {"name": "Property RSF", "description": "Square footage."}}}
    )
    sheet = Sheet(name="S1", rows=((Cell("Property RSF", "A5"), Cell(339050, "B5")),))
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    fr = det.entries[0]["property_rsf"]
    assert fr.value == 339050
    assert fr.source_locations == ["S1!B5"]
    assert fr.citation == "[Sheet: S1, Cell: B5] 339050"


def test_quoted_description_label_matches():
    schema = normalize_schema(
        {"fields": {"widget_code": {"name": "Widget Code", "description": "From the 'Widget Code' column, e.g. W-1."}}}
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Widget Code", "A1"), Cell("Count", "B1")),
            (Cell("Blue Widget", "A2"), Cell(4, "B2")),
        ),
    )
    doc = ParsedDocument(kind="csv", filename="s.csv", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.doc_level["widget_code"].value == "Blue Widget"
    assert det.doc_level["widget_code"].source_locations == ["S!A2"]


def test_mixed_alphanumeric_labelled_values_left_to_llm():
    schema = normalize_schema(
        {"fields": {"widget_code": {"name": "Widget Code", "description": "The widget code."}}}
    )
    sheet = Sheet(
        name="S",
        rows=((Cell("Widget Code", "A1"), Cell("W-9", "B1")),),
    )
    doc = ParsedDocument(kind="csv", filename="s.csv", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.doc_level.get("widget_code") is None
    assert det.entries[0]["widget_code"].value is None


def _table_schema():
    return normalize_schema(
        {
            "name": "units",
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "name": {"name": "Name", "description": "The 'Name' column."},
                "units": {"name": "Units", "description": "The 'Units' column."},
                "property_name": {"name": "Property", "description": "Property from the header."},
            },
        }
    )


def _table_doc():
    sheet = Sheet(
        name="Sheet1",
        rows=(
            (Cell("Property", "A1"), Cell("ACME Court", "B1")),
            (Cell(None, "A2"), Cell(None, "B2")),
            (Cell("Code", "A3"), Cell("Name", "B3"), Cell("Units", "C3")),
            (Cell("PR - P1", "A4"), Cell("P1 1x1", "B4"), Cell(24, "C4")),
            (Cell("PR - P2", "A5"), Cell("P2 2x2", "B5"), Cell(16, "C5")),
            (Cell("Totals", "A6"), Cell("", "B6"), Cell(40, "C6")),
        ),
    )
    return ParsedDocument(kind="xlsx", filename="t.xlsx", sheets=[sheet])


def test_table_mode_one_entry_per_row_skipping_totals():
    det = deterministic_extract(_table_schema(), _table_doc())
    assert det.mode == "table"
    assert len(det.entries) == 2
    first = det.entries[0]
    assert first["code"].value == "PR - P1"
    assert first["units"].value == 24
    assert first["code"].source_locations == ["Sheet1!A4"]
    assert first["name"].citation == "[Sheet: Sheet1, Cell: B4] P1 1x1"
    assert det.entries[1]["code"].value == "PR - P2"


def test_table_doc_level_overlay():
    det = deterministic_extract(_table_schema(), _table_doc())
    for entry in det.entries:
        assert entry["property_name"].value == "ACME Court"
        assert entry["property_name"].citation == "[Sheet: Sheet1, Cell: B1] ACME Court"


def test_table_column_collision_prefers_stronger_match():
    schema = normalize_schema(
        {
            "fields": {
                "market_rent": {"name": "Market Rent", "description": ""},
                "other": {"name": "Rent", "description": "The 'Rent' column."},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Market Rent", "A1"), Cell("Notes", "B1")),
            (Cell(1425, "A2"), Cell("x", "B2")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="c.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["market_rent"].value == 1425
    assert det.entries[0]["other"].value is None


def test_generic_widget_schema_on_xlsx_file():
    schema = normalize_schema(
        {
            "name": "gadgets",
            "fields": {
                "gadget_id": {"name": "Gadget", "description": "The 'Gadget' column.", "primary": True},
                "gadget_weight": {"name": "Weight", "description": "The 'Weight' column."},
            },
        }
    )
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["Gadget", "Weight"])
    ws.append(["G-1", 12.5])
    ws.append(["G-2", 8])
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "g.xlsx"
        wb.save(path)
        doc = parse_file(path)
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert det.entries[0]["gadget_id"].value == "G-1"
    assert det.entries[0]["gadget_weight"].value == 12.5
    assert det.entries[1]["gadget_weight"].value == 8


def test_header_without_primary_column_aborts_table_mode():
    schema = normalize_schema(
        {
            "fields": {
                "a": {"name": "Alpha", "description": ""},
                "b": {"name": "Beta", "description": ""},
            }
        }
    )
    doc = _table_doc()
    det = deterministic_extract(schema, doc)
    assert det.mode == "single"


def _multi_sheet_doc(sheets):
    return ParsedDocument(kind="xlsx", filename="wb.xlsx", sheets=sheets)


def test_auxiliary_parameter_sheet_does_not_poison_doc_level_fields():
    schema = normalize_schema(
        {
            "name": "units",
            "fields": {
                "unit": {"name": "Unit", "description": "The 'Unit' column.", "primary": True},
                "status": {"name": "Status", "description": "The 'Status' column."},
                "report_period": {"name": "Report Period", "description": "The month and year, as YYYY-MM."},
            },
        }
    )
    data = Sheet(
        name="Willow Data",
        rows=(
            (Cell("Unit", "A1"), Cell("Status", "B1")),
            (Cell("100-101", "A2"), Cell("Occupied", "B2")),
            (Cell("100-102", "A3"), Cell("Vacant", "B3")),
        ),
    )
    parameters = Sheet(
        name="Report Parameters",
        rows=((Cell("Report Period", "B7"), Cell("Prior Post Month", "C7")),),
    )
    det = deterministic_extract(schema, _multi_sheet_doc([data, parameters]))
    assert det.mode == "table"
    assert len(det.entries) == 2
    for entry in det.entries:
        assert entry["unit"].value in ("100-101", "100-102")
        assert entry["report_period"].value is None
        assert entry["report_period"].citation == ""


def test_conflicting_sheets_without_clear_source_stay_unresolved():
    schema = normalize_schema({"fields": {"alpha": {"name": "Alpha", "description": ""}}})
    s1 = Sheet(name="One", rows=((Cell("Alpha", "A1"), Cell("one", "B1")),))
    s2 = Sheet(name="Two", rows=((Cell("Alpha", "A1"), Cell("two", "B1")),))
    det = deterministic_extract(schema, _multi_sheet_doc([s1, s2]))
    assert det.entries[0]["alpha"].value is None


def test_clear_source_beats_lower_signal_sheet():
    schema = normalize_schema(
        {
            "fields": {
                "alpha": {"name": "Alpha", "description": ""},
                "beta": {"name": "Beta", "description": ""},
            }
        }
    )
    data = Sheet(
        name="Data",
        rows=(
            (Cell("Alpha", "A1"), Cell("from data", "B1")),
            (Cell("Beta", "A2"), Cell("beta value", "B2")),
        ),
    )
    notes = Sheet(name="Notes", rows=((Cell("Alpha", "A1"), Cell("from notes", "B1")),))
    det = deterministic_extract(schema, _multi_sheet_doc([data, notes]))
    fr = det.entries[0]["alpha"]
    assert fr.value == "from data"
    assert fr.source_locations == ["Data!B1"]
    assert det.entries[0]["beta"].value == "beta value"


def test_relevant_auxiliary_sheet_fills_gaps():
    schema = normalize_schema(
        {
            "name": "units",
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "units_n": {"name": "Units", "description": "The 'Units' column."},
                "owner": {"name": "Owner", "description": "The 'Owner' line."},
                "custodian": {"name": "Custodian", "description": "The 'Custodian' line."},
            },
        }
    )
    data = Sheet(
        name="Data",
        rows=(
            (Cell("Code", "A1"), Cell("Units", "B1")),
            (Cell("A-1", "A2"), Cell(5, "B2")),
        ),
    )
    info = Sheet(
        name="Info",
        rows=(
            (Cell("Owner", "A1"), Cell("ACME", "B1")),
            (Cell("Custodian", "A2"), Cell("Bob", "B2")),
        ),
    )
    det = deterministic_extract(schema, _multi_sheet_doc([data, info]))
    assert det.mode == "table"
    entry = det.entries[0]
    assert entry["owner"].value == "ACME"
    assert entry["owner"].source_locations == ["Info!B1"]
    assert entry["custodian"].value == "Bob"
    assert entry["code"].value == "A-1"
    assert entry["code"].source_locations == ["Data!A2"]


def test_single_label_auxiliary_sheet_is_ignored():
    schema = normalize_schema(
        {
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "units_n": {"name": "Units", "description": "The 'Units' column."},
                "owner": {"name": "Owner", "description": "The 'Owner' line."},
            }
        }
    )
    data = Sheet(
        name="Data",
        rows=(
            (Cell("Code", "A1"), Cell("Units", "B1")),
            (Cell("A-1", "A2"), Cell(5, "B2")),
        ),
    )
    info = Sheet(name="Info", rows=((Cell("Owner", "A1"), Cell("ACME", "B1")),))
    det = deterministic_extract(schema, _multi_sheet_doc([data, info]))
    assert det.entries[0]["owner"].value is None


def test_single_sheet_extraction_unchanged_by_pooling():
    schema = normalize_schema(
        {"fields": {"property_rsf": {"name": "Property RSF", "description": "Square footage."}}}
    )
    sheet = Sheet(name="S1", rows=((Cell("Property RSF", "A5"), Cell(339050, "B5")),))
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["property_rsf"].value == 339050


def _row_schema():
    return normalize_schema(
        {
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "qty": {"name": "Qty", "description": "The 'Qty' column."},
            }
        }
    )


def test_summary_row_variants_skipped():
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell("A-1", "A2"), Cell(3, "B2")),
            (Cell("Subtotal", "A3"), Cell(3, "B3")),
            (Cell("Grand Total", "A4"), Cell(30, "B4")),
            (Cell("Average", "A5"), Cell(3.5, "B5")),
            (Cell("A-2", "A6"), Cell(4, "B6")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(_row_schema(), doc)
    assert det.mode == "table"
    assert [e["code"].value for e in det.entries] == ["A-1", "A-2"]


def test_blank_rows_interleaved_are_skipped():
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell(None, "A2"), Cell(None, "B2")),
            (Cell("A-1", "A3"), Cell(3, "B3")),
            (Cell("", "A4"), Cell("", "B4")),
            (Cell("A-2", "A5"), Cell(4, "B5")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(_row_schema(), doc)
    assert det.mode == "table"
    assert [e["code"].value for e in det.entries] == ["A-1", "A-2"]


def test_charge_summary_rows_with_duplicate_primary_dropped():
    schema = normalize_schema(
        {
            "fields": {
                "unit": {"name": "Unit", "description": "The 'Unit' column.", "primary": True},
                "status": {"name": "Status", "description": "The 'Status' column."},
                "charge": {"name": "Charge", "description": "The 'Charge' column."},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Unit", "A1"), Cell("Status", "B1"), Cell("Charge", "C1")),
            (Cell("U-1", "A2"), Cell("Occupied", "B2"), Cell(None, "C2")),
            (Cell("U-1", "A3"), Cell(None, "B3"), Cell("Rent 900", "C3")),
            (Cell("U-1", "A4"), Cell(None, "B4"), Cell("Pet 25", "C4")),
            (Cell("U-2", "A5"), Cell("Vacant", "B5"), Cell(None, "C5")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert len(det.entries) == 2
    assert det.entries[0]["unit"].value == "U-1"
    assert det.entries[0]["status"].value == "Occupied"
    assert det.entries[0]["status"].source_locations == ["S!B2"]
    assert det.entries[1]["unit"].value == "U-2"
    assert any("duplicate primary" in w for w in det.warnings)


def test_candidate_with_solid_rows_beats_one_with_more_columns():
    schema = normalize_schema(
        {
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "qty": {"name": "Qty", "description": "The 'Qty' column."},
                "note": {"name": "Note", "description": "The 'Note' column."},
            }
        }
    )
    greedy = Sheet(
        name="Greedy",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1"), Cell("Note", "C1")),
            (Cell("X-1", "A2"), Cell(None, "B2"), Cell(None, "C2")),
        ),
    )
    solid_rows = [(Cell("Code", "A1"), Cell("Qty", "B1"))]
    for i in range(2, 52):
        solid_rows.append((Cell(f"S-{i}", f"A{i}"), Cell(i, f"B{i}")))
    solid = Sheet(name="Solid", rows=tuple(solid_rows))
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[greedy, solid])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert len(det.entries) == 50
    assert det.entries[0]["code"].citation.startswith("[Sheet: Solid,")
    assert det.entries[0]["code"].source_locations == ["Solid!A2"]


def test_duplicate_primary_identifiers_keep_first():
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell("A-1", "A2"), Cell(3, "B2")),
            (Cell("A-1", "A3"), Cell(99, "B3")),
            (Cell("A-2", "A4"), Cell(4, "B4")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(_row_schema(), doc)
    assert det.mode == "table"
    assert len(det.entries) == 2
    assert det.entries[0]["qty"].value == 3
    assert any("duplicate primary row(s) skipped" in w for w in det.warnings)
    assert not any(e["qty"].value == 99 for e in det.entries)


def test_legitimate_record_resembling_summary_row_is_kept():
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell("A-1", "A2"), Cell(3, "B2")),
            (Cell("Total Storage", "A3"), Cell(7, "B3")),
            (Cell("Subtotals", "A4"), Cell(10, "B4")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(_row_schema(), doc)
    assert det.mode == "table"
    assert [e["code"].value for e in det.entries] == ["A-1", "Total Storage"]


def test_identical_table_sheets_are_ambiguous():
    rows = (
        (Cell("Code", "A1"), Cell("Qty", "B1")),
        (Cell("A-1", "A2"), Cell(3, "B2")),
        (Cell("A-2", "A3"), Cell(4, "B3")),
    )
    doc = ParsedDocument(
        kind="xlsx",
        filename="s.xlsx",
        sheets=[Sheet(name="Copy1", rows=rows), Sheet(name="Copy2", rows=rows)],
    )
    det = deterministic_extract(_row_schema(), doc)
    assert det.mode == "single"
    assert any("ambiguous" in w for w in det.warnings)


def test_table_sheet_interior_is_not_a_doc_level_source():
    schema = normalize_schema(
        {
            "name": "jobs",
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "finish_by": {"name": "Finish By", "description": "The 'Finish By' column."},
                "owner": {"name": "Owner", "description": "The report owner."},
            },
        }
    )
    sheet = Sheet(
        name="Data",
        rows=(
            (Cell("Owner", "A1"), Cell("ACME", "B1")),
            (Cell("Code", "A2"), Cell("Finish By", "B2"), Cell("Tracking Id", "C2")),
            (Cell("A-1", "A3"), Cell(20260105, "B3")),
            (Cell("A-2", "A4"), Cell(20260106, "B4")),
            (Cell("A-3", "A5"), Cell(20260107, "B5")),
            (Cell("A-4", "A6"), Cell(20260108, "B6")),
            (Cell("Status Summary", "A7"), Cell("All jobs tracked", "B7")),
            (Cell("Total Jobs", "A8"), Cell("4 of 4", "B8")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert len(det.entries) == 4
    assert "finish_by" not in det.doc_level
    assert det.doc_level["owner"].value == "ACME"
    assert det.doc_level["owner"].source_locations == ["Data!B1"]
    for entry in det.entries:
        assert entry["owner"].value == "ACME"
        assert entry["finish_by"].citation.startswith("[Sheet: Data, Cell: B")
        assert entry["finish_by"].value in (20260105, 20260106, 20260107, 20260108)


def test_docx_table_header_row_not_a_doc_level_source():
    import docx as docx_lib
    import tempfile

    schema = normalize_schema(
        {
            "fields": {
                "person_name": {"name": "Name", "description": "Person."},
                "person_age": {"name": "Age", "description": "Years."},
            }
        }
    )
    d = docx_lib.Document()
    d.add_paragraph("Roster for spring")
    t = d.add_table(rows=3, cols=2)
    t.cell(0, 0).text = "Name"
    t.cell(0, 1).text = "Age"
    t.cell(1, 0).text = "Alice"
    t.cell(1, 1).text = "30"
    t.cell(2, 0).text = "Bob"
    t.cell(2, 1).text = "41"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.docx"
        d.save(p)
        doc = parse_file(p)
    det = deterministic_extract(schema, doc)
    assert det.mode == "single"
    assert "person_name" not in det.doc_level
    assert det.entries[0]["person_name"].value is None
    assert det.entries[0]["person_age"].value is None


def test_competing_rent_and_market_rent_labels():
    schema = normalize_schema(
        {
            "fields": {
                "rent": {"name": "Rent", "description": "The base rent."},
                "market_rent": {"name": "Market Rent", "description": "The market rent."},
            }
        }
    )
    doc = _pdf_doc([["Rent: 1200", "Market Rent: 1500"]])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["rent"].value == 1200
    assert det.entries[0]["market_rent"].value == 1500
    assert det.entries[0]["rent"].citation == "[Page 1] Rent: 1200"
    assert det.entries[0]["market_rent"].citation == "[Page 1] Market Rent: 1500"


def test_competing_table_columns_rent_vs_market_rent():
    schema = normalize_schema(
        {
            "fields": {
                "item": {"name": "Item", "description": "The 'Item' column.", "primary": True},
                "rent": {"name": "Rent", "description": "Base rent."},
                "market_rent": {"name": "Market Rent", "description": "Market rent."},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Market Rent", "A1"), Cell("Rent", "B1"), Cell("Item", "C1")),
            (Cell(1500, "A2"), Cell(1200, "B2"), Cell("I-1", "C2")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert det.entries[0]["market_rent"].value == 1500
    assert det.entries[0]["rent"].value == 1200
    assert det.entries[0]["market_rent"].source_locations == ["S!A2"]
    assert det.entries[0]["rent"].source_locations == ["S!B2"]


def test_duplicate_field_names_contest_the_label():
    schema = normalize_schema(
        {
            "fields": {
                "alpha_one": {"name": "Alpha", "description": "First."},
                "alpha_two": {"name": "Alpha", "description": "Second."},
            }
        }
    )
    doc = _pdf_doc([["Alpha: contested value"]])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["alpha_one"].value is None
    assert det.entries[0]["alpha_two"].value is None
    assert any("matched multiple fields equally" in w for w in det.warnings)


def test_schema_order_does_not_change_matching():
    fields = {
        "rent": {"name": "Rent", "description": "Base rent."},
        "market_rent": {"name": "Market Rent", "description": "Market rent."},
        "total_rent": {"name": "Total Rent", "description": "Total."},
    }
    doc = _pdf_doc([["Rent: 1200", "Market Rent: 1500", "Total Rent: 18000"]])
    det_forward = deterministic_extract(normalize_schema({"fields": fields}), doc)
    det_reverse = deterministic_extract(
        normalize_schema({"fields": dict(reversed(list(fields.items())))}), doc
    )
    for det in (det_forward, det_reverse):
        entry = det.entries[0]
        assert entry["rent"].value == 1200
        assert entry["market_rent"].value == 1500
        assert entry["total_rent"].value == 18000


def test_exact_match_beats_weak_subset_match():
    schema = normalize_schema(
        {
            "fields": {
                "rent": {"name": "Rent", "description": "Base rent."},
                "total_rent": {"name": "Total Rent", "description": "Total rent over term."},
            }
        }
    )
    doc = _pdf_doc([["Total Rent: 18000", "Rent: 1200"]])
    det = deterministic_extract(schema, doc)
    assert det.entries[0]["rent"].value == 1200
    assert det.entries[0]["total_rent"].value == 18000


def test_equal_strength_competition_in_table_columns():
    schema = normalize_schema(
        {
            "fields": {
                "qty_one": {"name": "Qty", "description": "First quantity."},
                "qty_two": {"name": "Qty", "description": "Second quantity."},
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell("C-1", "A2"), Cell(3, "B2")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert det.entries[0]["code"].value == "C-1"
    assert det.entries[0]["qty_one"].value is None
    assert det.entries[0]["qty_two"].value is None


def test_quoted_label_beats_token_subset():
    schema = normalize_schema(
        {
            "fields": {
                "item": {"name": "Item", "description": "The 'Item' column.", "primary": True},
                "rent_code": {"name": "Code", "description": "The 'Rent Code' column."},
                "rent": {"name": "Rent", "description": "Base rent."},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("Item", "A1"), Cell("Rent Code", "B1"), Cell("Rent", "C1")),
            (Cell("I-1", "A2"), Cell("RC-1", "B2"), Cell(900, "C2")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    det = deterministic_extract(schema, doc)
    assert det.mode == "table"
    assert det.entries[0]["rent_code"].value == "RC-1"
    assert det.entries[0]["rent"].value == 900
    assert det.entries[0]["rent_code"].source_locations == ["S!B2"]
