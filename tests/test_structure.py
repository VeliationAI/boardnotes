from __future__ import annotations

from boardnotes.models import DiagramRegion, OcrLine
from boardnotes.structure import classify_line, lines_to_blocks, structure_to_markdown


def test_heading_bullet_numbered():
    lines = [
        OcrLine("Sprint Planning", (10, 10, 200, 40)),
        OcrLine("Goals", (10, 60, 80, 32)),
        OcrLine("• Ship demo", (20, 100, 160, 24)),
        OcrLine("1. English first", (20, 140, 180, 24)),
        OcrLine("Pipeline: photo → OCR", (20, 180, 220, 22)),
    ]
    blocks = lines_to_blocks(lines)
    kinds = [b.kind for b in blocks]
    assert kinds[0] == "heading"
    assert "bullet" in kinds
    assert "numbered" in kinds
    assert "quote" in kinds

    md = structure_to_markdown(lines)
    assert md.startswith("# ")
    assert "- Ship demo" in md
    assert "1. English first" in md
    assert "> Pipeline:" in md


def test_diagram_section_in_markdown():
    lines = [OcrLine("Notes", (0, 0, 40, 20))]
    diagrams = [
        DiagramRegion(bbox=(0, 0, 10, 10), image_path="/tmp/out/diagrams/diagram_0.png")
    ]
    md = structure_to_markdown(lines, diagrams=diagrams)
    assert "## Diagrams" in md
    assert "diagrams/diagram_0.png" in md


def test_classify_bullet_variants():
    median = 24.0
    for prefix in ("• ", "- ", "* "):
        b = classify_line(OcrLine(f"{prefix}Item", (0, 0, 40, 20)), median, 3)
        assert b.kind == "bullet"
        assert b.text == "Item"
