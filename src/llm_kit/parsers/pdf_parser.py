# parser/pdf_parser.py

from pathlib import Path
from typing import Any, BinaryIO, cast

import pdfplumber

from .base import DocumentParser
from .models import ParsedChunk, ParsedDocument, ParsedSection


class PdfParser(DocumentParser):
    """
    Deterministic PDF parser.
    - Uses page order
    - Uses simple heading heuristics
    - Emits global character offsets
    """

    def parse(self, source: str | Path | BinaryIO) -> ParsedDocument:
        sections: list[ParsedSection] = []
        global_offset = 0

        current_heading = "Document"
        current_path = [current_heading]
        current_chunks: list[ParsedChunk] = []

        # pdfplumber.open accepts path-like or buffer objects; cast to Any
        with pdfplumber.open(cast(Any, source)) as pdf:
            title = self._extract_title(pdf)

            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                lines = text.splitlines()

                for line in lines:
                    clean = line.strip()
                    if not clean:
                        global_offset += 1
                        continue

                    # Heading heuristic
                    if self._is_heading(clean):
                        if current_chunks:
                            sections.append(
                                ParsedSection(
                                    heading=current_heading,
                                    section_path=current_path,
                                    chunks=current_chunks,
                                )
                            )
                            current_chunks = []

                        current_heading = clean
                        current_path = [current_heading]
                        global_offset += len(clean) + 1
                        continue

                    start = global_offset
                    end = start + len(clean)

                    current_chunks.append(
                        ParsedChunk(
                            text=clean,
                            offset_start=start,
                            offset_end=end,
                            metadata={"page": page_number},
                        )
                    )

                    global_offset = end + 1

        if current_chunks:
            sections.append(
                ParsedSection(
                    heading=current_heading,
                    section_path=current_path,
                    chunks=current_chunks,
                )
            )

        return ParsedDocument(
            title=title,
            metadata={"source_type": "pdf"},
            sections=sections,
        )

    def _extract_title(self, pdf: Any) -> str:
        """
        Simple heuristic:
        - First non-empty line of first page
        """
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return "Untitled Document"

    def _is_heading(self, line: str) -> bool:
        """
        Very conservative heading heuristic.
        """
        if len(line) > 120:
            return False
        if line.isupper():
            return True
        return line.endswith(":")
