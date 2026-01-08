from pathlib import Path

from llm_kit.parsers.models import ParsedDocument
from llm_kit.parsers.pdf_parser import PdfParser

# --- Document-level tests ---


def test_extracts_title(parsed_sample: ParsedDocument) -> None:
    assert parsed_sample.title == "SAMPLE DOCUMENT TITLE"


def test_sets_pdf_metadata(parsed_sample: ParsedDocument) -> None:
    assert parsed_sample.metadata == {"source_type": "pdf"}


def test_extracts_section_headings(parsed_sample: ParsedDocument) -> None:
    headings = [s.heading for s in parsed_sample.sections]
    assert headings == ["INTRODUCTION:", "DETAILS:", "CONCLUSION:"]


# --- Chunk-level tests ---


def test_blocks_have_correct_offset_math(parsed_sample: ParsedDocument) -> None:
    for section in parsed_sample.sections:
        for block in section.blocks:
            assert block.offset_end - block.offset_start == len(block.text)


def test_block_offsets_are_non_overlapping(parsed_sample: ParsedDocument) -> None:
    all_blocks = [c for s in parsed_sample.sections for c in s.blocks]

    for i in range(1, len(all_blocks)):
        assert all_blocks[i].offset_start > all_blocks[i - 1].offset_end


def test_blocks_have_page_metadata(parsed_sample: ParsedDocument) -> None:
    for section in parsed_sample.sections:
        for block in section.blocks:
            assert block.metadata["page"] == 1


def test_extracts_expected_content(parsed_sample: ParsedDocument) -> None:
    texts = [c.text for s in parsed_sample.sections for c in s.blocks]
    assert "This is the first paragraph of the introduction." in texts
    assert "Some identifiers like user_id and order_id appear here." in texts


# --- Determinism test ---


def test_parsing_is_deterministic(pdf_dir: Path) -> None:
    parser = PdfParser()
    path = pdf_dir / "sample.pdf"

    with open(path, "rb") as f:
        first = parser.parse(f)
    with open(path, "rb") as f:
        second = parser.parse(f)

    assert first.title == second.title
    assert first.metadata == second.metadata
    assert len(first.sections) == len(second.sections)

    for s1, s2 in zip(first.sections, second.sections, strict=False):
        assert s1.heading == s2.heading
        assert len(s1.blocks) == len(s2.blocks)
        for c1, c2 in zip(s1.blocks, s2.blocks, strict=False):
            assert (c1.text, c1.offset_start, c1.offset_end) == (
                c2.text,
                c2.offset_start,
                c2.offset_end,
            )


# --- Multi-page tests ---


def test_multipage_extracts_blocks_from_multiple_pages(
    parsed_multipage: ParsedDocument,
) -> None:
    pages = {c.metadata["page"] for s in parsed_multipage.sections for c in s.blocks}
    assert pages == {1, 2}


def test_multipage_page_numbers_are_correct(parsed_multipage: ParsedDocument) -> None:
    all_blocks = [c for s in parsed_multipage.sections for c in s.blocks]

    page1_texts = [c.text for c in all_blocks if c.metadata["page"] == 1]
    page2_texts = [c.text for c in all_blocks if c.metadata["page"] == 2]

    assert "This content is on page one." in page1_texts
    assert "This content is on page two." in page2_texts


# --- Heading heuristic tests ---


def test_all_caps_line_is_heading(parsed_edge_case: ParsedDocument) -> None:
    headings = [s.heading for s in parsed_edge_case.sections]
    assert "ALL CAPS HEADING" in headings


def test_line_with_colon_is_heading(parsed_edge_case: ParsedDocument) -> None:
    headings = [s.heading for s in parsed_edge_case.sections]
    assert "HEADING WITH COLON:" in headings


def test_long_line_is_not_heading(parsed_edge_case: ParsedDocument) -> None:
    headings = [s.heading for s in parsed_edge_case.sections]
    # No heading should be > 120 chars
    assert all(len(h) <= 120 for h in headings)

    # The long line should appear in block text, not as heading
    all_texts = [c.text for s in parsed_edge_case.sections for c in s.blocks]
    assert any(len(t) > 120 for t in all_texts)
