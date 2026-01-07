from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from llm_kit.parsers.models import ParsedDocument
from llm_kit.parsers.pdf_parser import PdfParser


def _create_sample_pdf(path: Path) -> None:
    """Creates a deterministic single-page PDF for integration testing."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    text = c.beginText(40, height - 50)

    lines = [
        "SAMPLE DOCUMENT TITLE",
        "",
        "INTRODUCTION:",
        "This is the first paragraph of the introduction.",
        "This is the second paragraph of the introduction.",
        "",
        "DETAILS:",
        "Here we describe details.",
        "Some identifiers like user_id and order_id appear here.",
        "",
        "CONCLUSION:",
        "This is the final section.",
    ]

    for line in lines:
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()


def _create_multipage_pdf(path: Path) -> None:
    """Creates a deterministic multi-page PDF for integration testing."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    # Page 1
    text = c.beginText(40, height - 50)
    page1_lines = [
        "MULTIPAGE DOCUMENT",
        "",
        "PAGE ONE CONTENT:",
        "This content is on page one.",
    ]
    for line in page1_lines:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    # Page 2
    text = c.beginText(40, height - 50)
    page2_lines = [
        "PAGE TWO CONTENT:",
        "This content is on page two.",
    ]
    for line in page2_lines:
        text.textLine(line)
    c.drawText(text)
    c.showPage()

    c.save()


def _create_edge_case_pdf(path: Path) -> None:
    """
    Creates a PDF with edge cases for heading heuristic testing.
    - Long line (>120 chars) should NOT be a heading
    - All caps line IS a heading
    - Line ending with colon IS a heading
    """
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    text = c.beginText(40, height - 50)

    long_line = "A" * 130  # > 120 chars, should NOT be heading

    lines = [
        "EDGE CASE DOCUMENT",
        "",
        "ALL CAPS HEADING",
        "Normal paragraph text here.",
        "",
        "HEADING WITH COLON:",
        "More normal text.",
        "",
        long_line,
        "Text after long line.",
    ]

    for line in lines:
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()


@pytest.fixture(scope="module")
def pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create all test PDFs once per module."""
    dir_path: Path = tmp_path_factory.mktemp("pdfs")

    _create_sample_pdf(dir_path / "sample.pdf")
    _create_multipage_pdf(dir_path / "multipage.pdf")
    _create_edge_case_pdf(dir_path / "edge_case.pdf")

    return dir_path


@pytest.fixture(scope="module")
def parsed_sample(pdf_dir: Path) -> ParsedDocument:
    """Parse sample PDF once, reuse across tests."""
    parser = PdfParser()
    with open(pdf_dir / "sample.pdf", "rb") as f:
        return parser.parse(f)


@pytest.fixture(scope="module")
def parsed_multipage(pdf_dir: Path) -> ParsedDocument:
    """Parse multipage PDF once, reuse across tests."""
    parser = PdfParser()
    with open(pdf_dir / "multipage.pdf", "rb") as f:
        return parser.parse(f)


@pytest.fixture(scope="module")
def parsed_edge_case(pdf_dir: Path) -> ParsedDocument:
    """Parse edge case PDF once, reuse across tests."""
    parser = PdfParser()
    with open(pdf_dir / "edge_case.pdf", "rb") as f:
        return parser.parse(f)
