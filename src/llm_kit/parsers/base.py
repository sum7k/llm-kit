# parser/base.py

from abc import ABC, abstractmethod
from typing import BinaryIO

from .models import ParsedDocument


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, source: BinaryIO) -> ParsedDocument:
        """
        Parse a document and return a structured, deterministic representation.

        Requirements:
        - Deterministic output for same input
        - Offsets are document-global
        - No IDs generated
        """
        raise NotImplementedError
