from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OcrLine:
    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 1.0


@dataclass
class DiagramRegion:
    bbox: tuple[int, int, int, int]
    image_path: str
    label: str = "diagram"


@dataclass
class ConvertResult:
    file_name: str
    width: int
    height: int
    ocr_backend: str
    warped: bool
    original_path: str | None = None
    rectified_path: str | None = None
    cleaned_path: str | None = None
    markdown: str = ""
    markdown_path: str | None = None
    pdf_path: str | None = None
    lines: list[OcrLine] = field(default_factory=list)
    diagrams: list[DiagramRegion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "ocr_backend": self.ocr_backend,
            "warped": self.warped,
            "markdown": self.markdown,
            "line_count": len(self.lines),
            "diagram_count": len(self.diagrams),
            "notes": self.notes,
            "diagrams": [
                {"bbox": list(d.bbox), "image_path": d.image_path, "label": d.label}
                for d in self.diagrams
            ],
        }
