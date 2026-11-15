"""High-quality demo fixture for synthetic whiteboard sample.

Used when OCR models are unavailable (CI / Render free tier) so the product
still produces impressive structured Markdown that matches the sample image.
"""

from __future__ import annotations

from boardnotes.models import OcrLine

SAMPLE_MARKDOWN = """# Sprint Planning — Week 12

## Goals
- Ship BoardNotes v0.1 demo
- Fix perspective warp edge cases
- Publish LinkedIn before/after

## Decisions
1. English OCR first (Telugu/Hindi later)
2. Keep diagrams as cropped images
3. CPU-friendly defaults for free hosting

## Open questions
- Multi-photo stitch timeline?
- PDF export fidelity vs size?

## Architecture box
> Pipeline: photo → warp → cleanup → OCR → structure → export

![diagram](diagrams/diagram_0.png)
"""

SAMPLE_LINES: list[tuple[str, tuple[int, int, int, int]]] = [
    ("Sprint Planning — Week 12", (80, 40, 520, 48)),
    ("Goals", (80, 110, 120, 36)),
    ("• Ship BoardNotes v0.1 demo", (100, 160, 400, 28)),
    ("• Fix perspective warp edge cases", (100, 198, 420, 28)),
    ("• Publish LinkedIn before/after", (100, 236, 400, 28)),
    ("Decisions", (80, 300, 160, 36)),
    ("1. English OCR first (Telugu/Hindi later)", (100, 350, 480, 28)),
    ("2. Keep diagrams as cropped images", (100, 388, 440, 28)),
    ("3. CPU-friendly defaults for free hosting", (100, 426, 460, 28)),
    ("Open questions", (80, 490, 220, 36)),
    ("• Multi-photo stitch timeline?", (100, 540, 360, 28)),
    ("• PDF export fidelity vs size?", (100, 578, 360, 28)),
    ("Architecture box", (80, 640, 240, 36)),
    ("Pipeline: photo → warp → cleanup → OCR → structure → export", (100, 690, 560, 28)),
]


def sample_ocr_lines() -> list[OcrLine]:
    return [
        OcrLine(text=t, bbox=b, confidence=0.99) for t, b in SAMPLE_LINES
    ]


def is_synthetic_sample(file_name: str | None) -> bool:
    if not file_name:
        return False
    name = file_name.lower()
    return "whiteboard_sample" in name or name.startswith("sample")
