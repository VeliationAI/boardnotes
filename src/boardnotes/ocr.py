"""Handwriting / board OCR with graceful fallbacks.

Priority (BOARDNOTES_OCR_BACKEND=auto):
  1. TrOCR (optional, heavy — torch/transformers)
  2. EasyOCR (optional)
  3. Tesseract (optional system binary)
  4. Deterministic mock / demo fixture

Env:
  BOARDNOTES_OCR_BACKEND = auto | trocr | easyocr | tesseract | mock
  BOARDNOTES_LIGHTWEIGHT = 1  → skip heavy models (Render free tier)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

import cv2
import numpy as np

from boardnotes.demo_fixture import sample_ocr_lines
from boardnotes.models import OcrLine

logger = logging.getLogger(__name__)


def _env_backend() -> str:
    return os.environ.get("BOARDNOTES_OCR_BACKEND", "auto").strip().lower()


def _lightweight() -> bool:
    return os.environ.get("BOARDNOTES_LIGHTWEIGHT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _line_bbox_from_boxes(boxes: list) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for b in boxes:
        if isinstance(b, (list, tuple)) and len(b) >= 2:
            xs.append(int(b[0]))
            ys.append(int(b[1]))
    if not xs:
        return (0, 0, 1, 1)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _ocr_tesseract(image: np.ndarray) -> list[OcrLine] | None:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    try:
        data = pytesseract.image_to_data(gray, output_type=Output.DICT, config="--psm 6")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract failed: %s", exc)
        return None

    lines: dict[tuple[int, int], list[tuple[int, int, int, int, str, float]]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw) / 100.0 if float(conf_raw) >= 0 else 0.5
        except (TypeError, ValueError):
            conf = 0.5
        key = (int(data["block_num"][i]), int(data["line_num"][i]))
        box = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        lines.setdefault(key, []).append((*box, text, conf))

    out: list[OcrLine] = []
    for key in sorted(lines.keys(), key=lambda k: (lines[k][0][1], lines[k][0][0])):
        parts = lines[key]
        x0 = min(p[0] for p in parts)
        y0 = min(p[1] for p in parts)
        x1 = max(p[0] + p[2] for p in parts)
        y1 = max(p[1] + p[3] for p in parts)
        text = " ".join(p[4] for p in parts)
        conf = float(np.mean([p[5] for p in parts]))
        out.append(OcrLine(text=text, bbox=(x0, y0, x1 - x0, y1 - y0), confidence=conf))
    return out if out else None


def _ocr_easyocr(image: np.ndarray) -> list[OcrLine] | None:
    if _lightweight():
        return None
    try:
        import easyocr
    except ImportError:
        return None

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EasyOCR failed: %s", exc)
        return None

    out: list[OcrLine] = []
    for item in results:
        box, text, conf = item[0], item[1], float(item[2])
        if not (text or "").strip():
            continue
        out.append(
            OcrLine(
                text=text.strip(),
                bbox=_line_bbox_from_boxes(box),
                confidence=conf,
            )
        )
    out.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return out if out else None


def _ocr_trocr(image: np.ndarray) -> list[OcrLine] | None:
    if _lightweight():
        return None
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
    except ImportError:
        return None

    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Split into horizontal bands as crude line detection
        h = gray.shape[0]
        band_h = max(40, h // 16)
        out: list[OcrLine] = []
        for y in range(0, h, band_h):
            band = image[y : min(h, y + band_h)]
            if band.size == 0:
                continue
            # Skip near-empty bands
            if float(np.mean(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY))) > 245:
                continue
            pil = Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
            pixel_values = processor(images=pil, return_tensors="pt").pixel_values
            with torch.no_grad():
                ids = model.generate(pixel_values)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:
                out.append(
                    OcrLine(
                        text=text,
                        bbox=(0, y, image.shape[1], band.shape[0]),
                        confidence=0.85,
                    )
                )
        return out if out else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("TrOCR failed: %s", exc)
        return None


def _mock_ocr(image: np.ndarray, prefer_fixture: bool = False) -> list[OcrLine]:
    """Deterministic OCR for CI / demo when no engine is available."""
    if prefer_fixture:
        return sample_ocr_lines()

    # Heuristic: if image looks like our synthetic board (green title bar area),
    # use the fixture; else emit a minimal placeholder from horizontal projection.
    h, w = image.shape[:2]
    if h > 400 and w > 500:
        # Check for dark ink density typical of our sample
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink = np.mean(gray < 80)
        if ink > 0.01:
            return sample_ocr_lines()

    return [
        OcrLine(
            text="[BoardNotes demo OCR — install EasyOCR or Tesseract for live recognition]",
            bbox=(20, 20, max(1, w - 40), 40),
            confidence=1.0,
        )
    ]


BackendFn = Callable[[np.ndarray], list[OcrLine] | None]


def resolve_backend(name: str | None = None) -> tuple[str, BackendFn]:
    """Pick OCR backend. Returns (name, callable)."""
    requested = (name or _env_backend()).lower()
    light = _lightweight()

    registry: list[tuple[str, BackendFn]] = []
    if not light:
        registry.append(("trocr", _ocr_trocr))
        registry.append(("easyocr", _ocr_easyocr))
    registry.append(("tesseract", _ocr_tesseract))

    if requested == "mock":
        return "mock", lambda img: _mock_ocr(img, prefer_fixture=True)

    if requested in {"trocr", "easyocr", "tesseract"}:
        fn = {
            "trocr": _ocr_trocr,
            "easyocr": _ocr_easyocr,
            "tesseract": _ocr_tesseract,
        }[requested]
        return requested, fn

    # auto
    for label, fn in registry:
        # Probe import cheaply without running full OCR
        if label == "trocr":
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
            except ImportError:
                continue
            return "trocr", fn
        if label == "easyocr":
            try:
                import easyocr  # noqa: F401
            except ImportError:
                continue
            return "easyocr", fn
        if label == "tesseract":
            try:
                import pytesseract  # noqa: F401
            except ImportError:
                continue
            return "tesseract", fn

    return "mock", lambda img: _mock_ocr(img, prefer_fixture=True)


def run_ocr(
    image: np.ndarray,
    backend: str | None = None,
    prefer_fixture: bool = False,
) -> tuple[list[OcrLine], str]:
    """Run OCR. Returns (lines, backend_used)."""
    if prefer_fixture or backend == "mock":
        return _mock_ocr(image, prefer_fixture=True), "mock"

    name, fn = resolve_backend(backend)
    if name == "mock":
        return _mock_ocr(image, prefer_fixture=prefer_fixture), "mock"

    result = fn(image)
    if result:
        return result, name

    # Fallback chain
    for fallback_name, fallback_fn in [
        ("easyocr", _ocr_easyocr),
        ("tesseract", _ocr_tesseract),
    ]:
        if fallback_name == name:
            continue
        if _lightweight() and fallback_name == "easyocr":
            continue
        got = fallback_fn(image)
        if got:
            return got, fallback_name

    return _mock_ocr(image, prefer_fixture=prefer_fixture), "mock"


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\u2022", "•").replace("·", "•")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text
