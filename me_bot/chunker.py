"""Markdown chunker.

Splits a corpus markdown file into ~400-token chunks at H2 boundaries,
falling back to paragraph splits when a section runs long. Each chunk
carries the frontmatter title, the section name, and a stable chunk_id
of the form `<relpath-without-ext>#<section>` for citation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    section: str
    text: str
    last_reviewed: str
    tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    _, fm, body = parts
    try:
        meta = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, body.strip()


def _split_long(text: str, max_chars: int = 1600) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    out: list[str] = []
    cur = ""
    for p in paragraphs:
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


def _slug_from_section(name: str) -> str:
    s = re.sub(r"[^a-z0-9\-]+", "-", name.strip().lower())
    return s.strip("-")[:48] or "section"


def chunk_markdown(path: Path, content: str, corpus_root: Path) -> list[Chunk]:
    meta, body = parse_frontmatter(content)
    title = meta.get("title", path.stem)
    last_reviewed = str(meta.get("last_reviewed", "1970-01-01"))
    tags = list(meta.get("tags", []) or [])

    rel = path.relative_to(corpus_root).as_posix().removesuffix(".md")

    sections = re.split(r"\n## ", body)
    chunks: list[Chunk] = []
    for i, section in enumerate(sections):
        if i == 0:
            section_name = "intro"
            section_text = section.strip()
        else:
            head, _, rest = section.partition("\n")
            section_name = _slug_from_section(head)
            section_text = rest.strip()
        if not section_text:
            continue
        for sub_idx, sub in enumerate(_split_long(section_text)):
            chunk_id = f"{rel}#{section_name}"
            if sub_idx > 0:
                chunk_id += f"-{sub_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_path=str(path),
                title=title,
                section=section_name,
                text=sub.strip(),
                last_reviewed=last_reviewed,
                tags=tags,
            ))
    return chunks
