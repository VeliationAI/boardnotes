from __future__ import annotations

import numpy as np

from boardnotes.cleanup import cleanup_board, enhance_contrast, reduce_glare, sharpen


def test_cleanup_preserves_shape():
    img = np.full((120, 160, 3), 210, dtype=np.uint8)
    img[40:80, 50:110] = 30  # ink blob
    out = cleanup_board(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_glare_and_contrast():
    img = np.full((80, 80, 3), 180, dtype=np.uint8)
    img[20:40, 20:40] = 250
    g = reduce_glare(img)
    c = enhance_contrast(g)
    s = sharpen(c)
    assert g.shape == img.shape
    assert c.shape == img.shape
    assert s.shape == img.shape
