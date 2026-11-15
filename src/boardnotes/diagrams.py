"""Detect non-text diagram regions, crop, and save as images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from boardnotes.models import DiagramRegion, OcrLine


def _text_mask(shape: tuple[int, ...], lines: list[OcrLine], pad: int = 8) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ln in lines:
        x, y, bw, bh = ln.bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        mask[y0:y1, x0:x1] = 255
    return mask


def detect_diagram_regions(
    image: np.ndarray,
    lines: list[OcrLine],
    min_area_ratio: float = 0.02,
    max_regions: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Find ink blobs that are not covered by OCR text boxes (likely diagrams)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Ink is dark on light board
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    text = _text_mask(image.shape, lines, pad=12)
    # Dilate text mask so nearby ink is excluded
    text = cv2.dilate(text, np.ones((9, 9), np.uint8), iterations=2)
    non_text = cv2.bitwise_and(ink, cv2.bitwise_not(text))

    contours, _ = cv2.findContours(non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    boxes: list[tuple[int, int, int, int, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # Skip full-width thin strokes (underlines)
        if bh < 40 and bw > w * 0.5:
            continue
        # Prefer boxy / square-ish diagram regions
        aspect = bw / max(bh, 1)
        if aspect > 6 or aspect < 0.15:
            continue
        boxes.append((x, y, bw, bh, area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, bw, bh) for x, y, bw, bh, _ in boxes[:max_regions]]


def crop_diagrams(
    image: np.ndarray,
    lines: list[OcrLine],
    out_dir: Path,
    pad: int = 16,
) -> list[DiagramRegion]:
    """Detect, crop, and write diagram PNGs under out_dir/diagrams/."""
    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(parents=True, exist_ok=True)

    boxes = detect_diagram_regions(image, lines)
    h, w = image.shape[:2]
    regions: list[DiagramRegion] = []

    for i, (x, y, bw, bh) in enumerate(boxes):
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        path = diag_dir / f"diagram_{i}.png"
        cv2.imwrite(str(path), crop)
        regions.append(
            DiagramRegion(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                image_path=str(path),
                label=f"diagram_{i}",
            )
        )
    return regions


def extract_forced_region(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_path: Path,
) -> DiagramRegion:
    """Crop a known region (used by sample generator / tests)."""
    x, y, bw, bh = bbox
    h, w = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    crop = image[y0:y1, x0:x1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return DiagramRegion(bbox=(x0, y0, x1 - x0, y1 - y0), image_path=str(out_path))
