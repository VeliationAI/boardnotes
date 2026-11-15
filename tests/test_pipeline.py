from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from boardnotes.pipeline import convert, save_image


@pytest.fixture
def skewed_photo(tmp_path: Path) -> Path:
    # Bright board on dark wall with text
    board = np.full((400, 600, 3), 235, dtype=np.uint8)
    cv2.putText(board, "Goals", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 40, 30), 3)
    cv2.putText(board, "- Ship demo", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 40, 30), 2)
    canvas = np.full((500, 720, 3), 45, dtype=np.uint8)
    src = np.float32([[0, 0], [599, 0], [599, 399], [0, 399]])
    dst = np.float32([[70, 40], [680, 60], [690, 460], [40, 430]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (720, 500), borderValue=(45, 45, 45))
    path = tmp_path / "shot.jpg"
    save_image(path, warped)
    return path


def test_convert_mock_ocr(skewed_photo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOARDNOTES_OCR_BACKEND", "mock")
    monkeypatch.setenv("BOARDNOTES_LIGHTWEIGHT", "1")
    out = tmp_path / "out"
    result = convert(skewed_photo, out_dir=out, ocr_backend="mock", pdf=True)
    assert result.ocr_backend == "mock"
    assert result.markdown_path and Path(result.markdown_path).exists()
    assert result.pdf_path and Path(result.pdf_path).exists()
    assert result.rectified_path and Path(result.rectified_path).exists()
    assert result.cleaned_path and Path(result.cleaned_path).exists()
    assert "# " in result.markdown or "Goals" in result.markdown or "BoardNotes" in result.markdown


def test_convert_synthetic_name_uses_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOARDNOTES_LIGHTWEIGHT", "1")
    img = np.full((300, 400, 3), 220, dtype=np.uint8)
    path = tmp_path / "whiteboard_sample.jpg"
    save_image(path, img)
    out = tmp_path / "out2"
    result = convert(path, out_dir=out, ocr_backend="mock", use_fixture=True, pdf=False)
    assert "Sprint Planning" in result.markdown
    assert result.markdown_path
