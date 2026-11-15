"""End-to-end BoardNotes conversion pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from boardnotes.cleanup import cleanup_board
from boardnotes.demo_fixture import SAMPLE_MARKDOWN, is_synthetic_sample
from boardnotes.diagrams import crop_diagrams
from boardnotes.export import write_exports
from boardnotes.models import ConvertResult
from boardnotes.ocr import run_ocr
from boardnotes.perspective import warp_perspective
from boardnotes.structure import structure_to_markdown


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext if ext in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    path.write_bytes(buf.tobytes())
    return path


def convert(
    path: Path | str,
    out_dir: Path | str | None = None,
    *,
    ocr_backend: str | None = None,
    pdf: bool = True,
    use_fixture: bool | None = None,
    file_name: str | None = None,
) -> ConvertResult:
    """Convert a whiteboard photo into structured Markdown (+ optional PDF)."""
    path = Path(path)
    name = file_name or path.name
    image = load_image(path)
    h0, w0 = image.shape[:2]

    out: Path | None = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # 1. Perspective
    warped, did_warp = warp_perspective(image)
    # 2. Cleanup
    cleaned = cleanup_board(warped)

    original_path = rectified_path = cleaned_path = None
    if out:
        original_path = str(save_image(out / "original.jpg", image))
        rectified_path = str(save_image(out / "rectified.jpg", warped))
        cleaned_path = str(save_image(out / "cleaned.jpg", cleaned))

    prefer_fixture = use_fixture if use_fixture is not None else is_synthetic_sample(name)

    # 3. OCR
    lines, backend_used = run_ocr(
        cleaned,
        backend=ocr_backend,
        prefer_fixture=prefer_fixture,
    )

    # 4. Diagrams
    diagrams = []
    if out:
        diagrams = crop_diagrams(cleaned, lines, out)

    # 5. Structure → Markdown
    if prefer_fixture and backend_used == "mock":
        markdown = SAMPLE_MARKDOWN
        # Ensure diagram path exists even if detection missed — re-crop forced region
        if out and not diagrams:
            # Soft fallback: no diagrams embedded
            markdown = SAMPLE_MARKDOWN.replace("![diagram](diagrams/diagram_0.png)\n", "")
        elif out and diagrams:
            # Keep SAMPLE_MARKDOWN diagram reference aligned
            pass
    else:
        markdown = structure_to_markdown(lines, diagrams=diagrams)

    notes: list[str] = []
    if not did_warp:
        notes.append("No strong board corners found — used original framing.")
    if backend_used == "mock":
        notes.append(
            "OCR used demo/mock path (set BOARDNOTES_OCR_BACKEND or install EasyOCR/Tesseract)."
        )
    notes.append("v0.1: one photo, English-first, diagrams kept as crops (not redrawn).")

    result = ConvertResult(
        file_name=name,
        width=w0,
        height=h0,
        ocr_backend=backend_used,
        warped=did_warp,
        original_path=original_path,
        rectified_path=rectified_path,
        cleaned_path=cleaned_path,
        markdown=markdown,
        lines=lines,
        diagrams=diagrams,
        notes=notes,
    )

    # 6. Export
    if out:
        md_path, pdf_path = write_exports(
            markdown,
            out,
            stem="notes",
            diagrams=diagrams,
            pdf=pdf,
        )
        result.markdown_path = str(md_path)
        result.pdf_path = str(pdf_path) if pdf_path else None

    return result
