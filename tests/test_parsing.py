from __future__ import annotations

import openpyxl
import pymupdf
import pytest

from app.parsing import ParseError, parse_file
from app.parsing.pdf import render_page


def _csv(tmp_path, text: str, name: str = "t.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["Code", "Rent"])
    ws.append(["A1", 1425])
    ws.append([None, None])
    ws2 = wb.create_sheet("Params")
    ws2["B2"] = 0.91
    p = tmp_path / "t.xlsx"
    wb.save(p)
    return p


def _docx(tmp_path):
    import docx as docx_lib

    d = docx_lib.Document()
    d.add_paragraph("First paragraph")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "H1"
    t.cell(0, 1).text = "H2"
    t.cell(1, 0).text = "a"
    t.cell(1, 1).text = "b"
    d.add_paragraph("Last paragraph")
    p = tmp_path / "t.docx"
    d.save(p)
    return p


def _pdf(tmp_path):
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Hello parsing layer")
    page2 = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10))
    page2.insert_image(page2.rect, pixmap=pix)
    p = tmp_path / "t.pdf"
    doc.save(str(p))
    doc.close()
    return p


class TestCsv:
    def test_grid_and_refs(self, tmp_path):
        doc = parse_file(_csv(tmp_path, "Unit,Rent\n101,1200\n102,1350\n"))
        assert doc.kind == "csv"
        assert doc.filename == "t.csv"
        sheet = doc.sheets[0]
        assert sheet.n_rows == 3 and sheet.n_cols == 2
        assert (sheet.rows[0][0].value, sheet.rows[0][0].ref) == ("Unit", "A1")
        assert (sheet.rows[1][1].value, sheet.rows[1][1].ref) == ("1200", "B2")

    def test_trailing_empty_cells_trimmed(self, tmp_path):
        doc = parse_file(_csv(tmp_path, "A,B,,,,\n1,2,,,,\n"))
        assert doc.sheets[0].n_cols == 2

    def test_quoted_comma_value(self, tmp_path):
        doc = parse_file(_csv(tmp_path, 'Name,Note\n"Doe, Jane",ok\n'))
        assert doc.sheets[0].rows[1][0].value == "Doe, Jane"

    def test_semicolon_delimiter_sniffed(self, tmp_path):
        doc = parse_file(_csv(tmp_path, "A;B\n1;2\n"))
        assert doc.sheets[0].rows[1][1].value == "2"

    def test_ref_columns_beyond_z(self, tmp_path):
        rows = [""] * 26 + ["X"]
        text = ",".join(rows) + "\n"
        doc = parse_file(_csv(tmp_path, text))
        assert doc.sheets[0].rows[0][25].ref == "Z1"
        assert doc.sheets[0].rows[0][26].ref == "AA1"


class TestXlsx:
    def test_sheets_values_and_refs(self, tmp_path):
        doc = parse_file(_xlsx(tmp_path))
        assert doc.kind == "xlsx"
        assert doc.filename == "t.xlsx"
        assert [s.name for s in doc.sheets] == ["Data", "Params"]
        data = doc.sheets[0]
        assert (data.rows[1][0].value, data.rows[1][0].ref) == ("A1", "A2")
        assert data.rows[1][1].value == 1425
        params = doc.sheets[1]
        assert (params.rows[1][1].value, params.rows[1][1].ref) == (0.91, "B2")

    def test_row_positions_preserved_with_blank_rows(self, tmp_path):
        doc = parse_file(_xlsx(tmp_path))
        assert doc.sheets[0].rows[2] == ()


class TestDocx:
    def test_sections_in_order(self, tmp_path):
        doc = parse_file(_docx(tmp_path))
        assert doc.kind == "docx"
        assert doc.filename == "t.docx"
        assert [s.kind for s in doc.sections] == ["paragraph", "table", "paragraph"]
        assert doc.sections[0].text == "First paragraph"
        table = doc.sections[1].table
        assert table.headers == ("H1", "H2")
        assert (table.rows[1][1].value, table.rows[1][1].ref) == ("b", "r2c2")
        assert doc.sections[2].text == "Last paragraph"


class TestPdf:
    def test_pages_and_scanned_detection(self, tmp_path):
        doc = parse_file(_pdf(tmp_path))
        assert doc.kind == "pdf"
        assert doc.filename == "t.pdf"
        assert len(doc.pages) == 2
        assert "Hello parsing layer" in doc.pages[0].text
        assert doc.scanned_pages == [2]
        assert any("text layer" in w for w in doc.warnings)

    def test_render_page_png(self, tmp_path):
        data = render_page(_pdf(tmp_path), 2, dpi=72)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_page_out_of_range(self, tmp_path):
        with pytest.raises(ParseError):
            render_page(_pdf(tmp_path), 99)


class TestDispatcher:
    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("x")
        with pytest.raises(ParseError, match="unsupported"):
            parse_file(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ParseError, match="not found"):
            parse_file(tmp_path / "nope.pdf")

    def test_extension_case_insensitive(self, tmp_path):
        assert parse_file(_csv(tmp_path, "A\n1\n", name="T.CSV")).kind == "csv"


class TestSamples:
    def test_every_sample_parses(self, samples_dir):
        for f in sorted(samples_dir.iterdir()):
            doc = parse_file(f)
            assert doc.kind in {"pdf", "docx", "xlsx", "csv"}
            assert doc.filename == f.name

    def test_lease_001_is_fully_scanned(self, samples_dir):
        doc = parse_file(samples_dir / "lease_001.pdf")
        assert len(doc.pages) == 25
        assert doc.scanned_pages == list(range(1, 26))

    def test_budget_has_text(self, samples_dir):
        doc = parse_file(samples_dir / "budget_001.pdf")
        assert not doc.scanned_pages
        assert sum(len(p.text) for p in doc.pages) > 500

    def test_lease_003_is_fully_scanned(self, samples_dir):
        doc = parse_file(samples_dir / "lease_003.pdf")
        assert len(doc.pages) == 21
        assert doc.scanned_pages == list(range(1, 22))

    def test_om_001_cover_page_flagged(self, samples_dir):
        doc = parse_file(samples_dir / "om_001.pdf")
        assert len(doc.pages) == 22
        assert doc.scanned_pages == [1]

    def test_t12_single_sheet(self, samples_dir):
        doc = parse_file(samples_dir / "t12_001.xlsx")
        assert [s.name for s in doc.sheets] == ["Report1"]
        assert doc.sheets[0].n_rows > 10

    def test_rentroll_two_sheets(self, samples_dir):
        doc = parse_file(samples_dir / "rentroll_001.xlsx")
        assert [s.name for s in doc.sheets] == ["Willow Creek Commons", "Report Parameters"]

    def test_boxscore_csv_grid(self, samples_dir):
        doc = parse_file(samples_dir / "boxscore_001.csv")
        sheet = doc.sheets[0]
        assert sheet.rows[1][0].value == "Pearl at Victory Ridge"
        refs = {c.ref for row in sheet.rows for c in row}
        assert "A7" in refs and "B7" in refs

    def test_lease_002_docx_paragraphs(self, samples_dir):
        doc = parse_file(samples_dir / "lease_002.docx")
        text = "\n".join(s.text for s in doc.sections if s.kind == "paragraph")
        assert "KESTREL" in text
        assert not any(s.kind == "table" for s in doc.sections)


def test_rentroll_params_sheet_padding_rows_do_not_hide_content(samples_dir):
    """The Report Parameters sheet declares 1000 rows but holds ~20 non-empty
    ones; parsing must keep the real content and skip padding rows."""
    doc = parse_file(samples_dir / "rentroll_001.xlsx")
    params = doc.sheets[1]
    assert params.name == "Report Parameters"
    non_empty = [row for row in params.rows if any(c.value not in (None, "") for c in row)]
    assert len(non_empty) == 20
    flat = {c.value for row in params.rows for c in row}
    assert "Rent Roll" in flat
    assert "Prior Post Month" in flat


def test_rentroll_data_sheet_preserves_datetime_cells(samples_dir):
    doc = parse_file(samples_dir / "rentroll_001.xlsx")
    data = doc.sheets[0]
    dt_values = [c.value for row in data.rows for c in row if hasattr(c.value, "year")]
    assert dt_values, "date cells should be preserved as date/datetime values"


def test_budget_pdf_tables_stay_in_text_layer_for_the_llm(samples_dir):
    """PyMuPDF find_tables mangles this document's borderless table headers;
    the parser must keep plain page text (with the GL header vocabulary)
    instead of pretending to detect table structure."""
    import pymupdf

    doc = parse_file(samples_dir / "budget_001.pdf")
    text = "\n".join(p.text for p in doc.pages)
    assert "2020 Budget" in text and "2021 Budget" in text
    assert "Lakeview" in text
    with pymupdf.open(samples_dir / "budget_001.pdf") as raw:
        tables = raw[0].find_tables().tables
        assert len(tables) >= 1
        extracted = " ".join(str(cell) for row in tables[0].extract() for cell in row)
        assert "2020Refore" in extracted
