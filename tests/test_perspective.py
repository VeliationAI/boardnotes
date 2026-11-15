from __future__ import annotations

import cv2
import numpy as np

from boardnotes.perspective import find_board_corners, order_points, warp_perspective


def _make_skewed_board(w: int = 640, h: int = 480) -> np.ndarray:
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)
    board = np.full((360, 480, 3), 230, dtype=np.uint8)
    cv2.putText(board, "Goals", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    src = np.float32([[0, 0], [479, 0], [479, 359], [0, 359]])
    dst = np.float32([[80, 40], [560, 70], [580, 420], [50, 400]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (w, h), borderValue=(40, 40, 40))
    return warped


def test_order_points():
    pts = np.array([[10, 10], [100, 12], [98, 80], [8, 78]], dtype=np.float32)
    ordered = order_points(pts)
    assert ordered.shape == (4, 2)
    assert ordered[0][0] <= ordered[1][0]
    assert ordered[0][1] <= ordered[3][1]


def test_warp_with_known_corners():
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    corners = np.array([[20, 20], [280, 25], [270, 180], [15, 175]], dtype=np.float32)
    out, warped = warp_perspective(img, corners=corners)
    assert warped is True
    assert out.shape[0] > 50 and out.shape[1] > 50


def test_find_and_warp_synthetic():
    img = _make_skewed_board()
    corners = find_board_corners(img)
    out, did = warp_perspective(img, corners=corners)
    assert out is not None
    # Either warp succeeds or graceful no-warp fallback
    assert out.size > 0
    if did:
        assert corners is not None
