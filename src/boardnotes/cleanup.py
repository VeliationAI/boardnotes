"""Image cleanup — glare reduction, contrast, sharpen."""

from __future__ import annotations

import cv2
import numpy as np


def reduce_glare(image: np.ndarray) -> np.ndarray:
    """Tone down specular highlights while preserving ink."""
    if image.ndim == 2:
        lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Soften very bright pixels
    bright = l > 230
    if np.any(bright):
        blurred = cv2.GaussianBlur(l, (21, 21), 0)
        l = np.where(bright, np.minimum(l, blurred), l).astype(np.uint8)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE contrast boost in LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray, amount: float = 1.2) -> np.ndarray:
    """Unsharp-mask style sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)


def binarize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Optional high-contrast binary for Tesseract-style OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def cleanup_board(image: np.ndarray, for_ocr: bool = False) -> np.ndarray:
    """Full cleanup pipeline for a rectified whiteboard image."""
    out = reduce_glare(image)
    out = enhance_contrast(out)
    out = denoise(out)
    out = sharpen(out, amount=0.9)
    if for_ocr:
        # Keep color cleaned version for display; OCR backends often prefer gray
        pass
    return out
