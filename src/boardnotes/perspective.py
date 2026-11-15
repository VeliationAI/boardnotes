"""Perspective correction — find whiteboard corners and warp flat."""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype("float32")
    # Fallback: min-area rectangle
    if len(approx) >= 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype("float32")
    return None


def find_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Detect the largest bright rectangular region (whiteboard)."""
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Bright-region threshold path (synthetic boards with dark borders)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: np.ndarray | None = None
    best_area = 0.0
    img_area = float(h * w)

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.15:
            continue
        if area > img_area * 0.98:
            continue
        quad = _quad_from_contour(c)
        if quad is None:
            continue
        if area > best_area:
            best_area = area
            best = quad

    if best is None:
        return None
    return order_points(best)


def warp_perspective(
    image: np.ndarray,
    corners: np.ndarray | None = None,
    max_width: int = 1600,
) -> tuple[np.ndarray, bool]:
    """Warp image flat using detected or provided corners.

    Returns (warped_image, did_warp).
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    pts = corners if corners is not None else find_board_corners(image)
    if pts is None:
        return image.copy(), False

    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = max(int(width_a), int(width_b), 1)
    max_h = max(int(height_a), int(height_b), 1)

    if max_w > max_width:
        scale = max_width / max_w
        max_w = max_width
        max_h = max(1, int(max_h * scale))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_w, max_h))
    return warped, True


def apply_known_corners(
    image: np.ndarray,
    corners: list[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    """Warp using explicitly provided corners (for tests / demos)."""
    pts = np.array(corners, dtype="float32")
    out, _ = warp_perspective(image, corners=pts)
    return out
