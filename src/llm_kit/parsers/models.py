# parser/models.py

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedChunk:
    text: str
    offset_start: int
    offset_end: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedSection:
    heading: str
    section_path: list[str]
    chunks: list[ParsedChunk]


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    metadata: dict
    sections: list[ParsedSection]
