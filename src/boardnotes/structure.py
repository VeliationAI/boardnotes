"""Structure detection — headings, bullets, numbering → Markdown hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from boardnotes.models import DiagramRegion, OcrLine
from boardnotes.ocr import normalize_ocr_text


BULLET_RE = re.compile(r"^[\u2022•\-\*\u25cf\u25e6]\s*(.+)$")
NUMBERED_RE = re.compile(r"^(\d+)[.)]\s*(.+)$")
HEADING_HINTS = re.compile(
    r"^(goals?|decisions?|open questions?|agenda|notes?|action items?|"
    r"architecture|summary|next steps?|todo|backlog)\b",
    re.I,
)


@dataclass
class Block:
    kind: str  # heading | bullet | numbered | quote | paragraph
    text: str
    level: int = 2
    number: int | None = None


def _median_height(lines: list[OcrLine]) -> float:
    if not lines:
        return 28.0
    heights = sorted(ln.bbox[3] for ln in lines)
    return float(heights[len(heights) // 2])


def classify_line(line: OcrLine, median_h: float, index: int) -> Block:
    text = normalize_ocr_text(line.text)
    if not text:
        return Block(kind="paragraph", text="")

    m_num = NUMBERED_RE.match(text)
    if m_num:
        return Block(kind="numbered", text=m_num.group(2).strip(), number=int(m_num.group(1)))

    m_bul = BULLET_RE.match(text)
    if m_bul:
        return Block(kind="bullet", text=m_bul.group(1).strip())

    # Quote / boxed note
    if text.startswith(">") or text.lower().startswith("pipeline:"):
        body = text[1:].strip() if text.startswith(">") else text
        return Block(kind="quote", text=body)

    word_count = len(text.split())
    is_short = len(text) < 48 and word_count <= 8
    tall = line.bbox[3] >= median_h * 1.2
    hint = bool(HEADING_HINTS.match(text))
    ends_question = text.endswith("?")

    if index == 0 and not ends_question:
        return Block(kind="heading", text=text.rstrip(":"), level=1)

    if hint or (tall and is_short and not ends_question):
        return Block(kind="heading", text=text.rstrip(":"), level=2)

    return Block(kind="paragraph", text=text)


def lines_to_blocks(lines: list[OcrLine]) -> list[Block]:
    median_h = _median_height(lines)
    blocks: list[Block] = []
    for i, ln in enumerate(lines):
        block = classify_line(ln, median_h, i)
        if block.text:
            blocks.append(block)
    return blocks


def blocks_to_markdown(
    blocks: list[Block],
    diagrams: list[DiagramRegion] | None = None,
) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.kind == "heading":
            hashes = "#" * max(1, min(b.level, 4))
            parts.append(f"{hashes} {b.text}")
            parts.append("")
        elif b.kind == "bullet":
            parts.append(f"- {b.text}")
        elif b.kind == "numbered":
            n = b.number or 1
            parts.append(f"{n}. {b.text}")
        elif b.kind == "quote":
            parts.append(f"> {b.text}")
            parts.append("")
        else:
            parts.append(b.text)
            parts.append("")

    if diagrams:
        if parts and parts[-1] != "":
            parts.append("")
        parts.append("## Diagrams")
        parts.append("")
        for i, d in enumerate(diagrams):
            rel = d.image_path.replace("\\", "/")
            # Prefer relative diagrams/ name in markdown
            if "diagrams/" in rel:
                rel = "diagrams/" + rel.split("diagrams/")[-1]
            elif "/" in rel or "\\" in rel:
                from pathlib import Path

                rel = f"diagrams/{Path(rel).name}"
            parts.append(f"![diagram {i + 1}]({rel})")
            parts.append("")

    # Collapse excess blank lines
    md = "\n".join(parts)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def structure_to_markdown(
    lines: list[OcrLine],
    diagrams: list[DiagramRegion] | None = None,
) -> str:
    return blocks_to_markdown(lines_to_blocks(lines), diagrams=diagrams)
