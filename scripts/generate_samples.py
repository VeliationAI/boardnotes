"""Generate synthetic whiteboard photos for demos (no real classroom photos)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            *candidates,
        ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_flat_board(width: int = 1200, height: int = 900) -> Image.Image:
    """Clean frontal whiteboard with structured ink content + a simple diagram."""
    img = Image.new("RGB", (width, height), (236, 238, 232))
    draw = ImageDraw.Draw(img)

    # Soft board frame
    draw.rectangle([18, 18, width - 19, height - 19], outline=(55, 72, 62), width=10)
    draw.rectangle([28, 28, width - 29, height - 29], outline=(180, 190, 178), width=2)

    # Faint grid / ghost marks
    rng = np.random.default_rng(42)
    for _ in range(40):
        x = int(rng.integers(60, width - 60))
        y = int(rng.integers(60, height - 60))
        draw.ellipse([x, y, x + 3, y + 3], fill=(210, 214, 205))

    title_f = _font(42, bold=True)
    h_f = _font(30, bold=True)
    body_f = _font(24)
    small_f = _font(20)

    ink = (28, 42, 36)
    accent = (40, 110, 88)

    draw.text((80, 55), "Sprint Planning — Week 12", font=title_f, fill=accent)

    y = 130
    draw.text((80, y), "Goals", font=h_f, fill=ink)
    y += 50
    for item in [
        "• Ship BoardNotes v0.1 demo",
        "• Fix perspective warp edge cases",
        "• Publish LinkedIn before/after",
    ]:
        draw.text((100, y), item, font=body_f, fill=ink)
        y += 38

    y += 24
    draw.text((80, y), "Decisions", font=h_f, fill=ink)
    y += 50
    for item in [
        "1. English OCR first (Telugu/Hindi later)",
        "2. Keep diagrams as cropped images",
        "3. CPU-friendly defaults for free hosting",
    ]:
        draw.text((100, y), item, font=body_f, fill=ink)
        y += 38

    y += 24
    draw.text((80, y), "Open questions", font=h_f, fill=ink)
    y += 50
    for item in [
        "• Multi-photo stitch timeline?",
        "• PDF export fidelity vs size?",
    ]:
        draw.text((100, y), item, font=body_f, fill=ink)
        y += 38

    y += 28
    draw.text((80, y), "Architecture box", font=h_f, fill=ink)
    y += 48
    # Diagram box (non-text region for crop detection)
    box = [90, y, 720, y + 150]
    draw.rounded_rectangle(box, radius=12, outline=accent, width=3)
    draw.rectangle([110, y + 30, 250, y + 110], outline=ink, width=2)
    draw.text((125, y + 55), "photo", font=small_f, fill=ink)
    # Arrows / nodes
    nodes = [(280, y + 70), (400, y + 70), (520, y + 70), (640, y + 70)]
    labels = ["warp", "clean", "OCR", "MD"]
    for (nx, ny), lab in zip(nodes, labels):
        draw.ellipse([nx - 28, ny - 28, nx + 28, ny + 28], outline=accent, width=2)
        draw.text((nx - 18, ny - 10), lab, font=small_f, fill=ink)
    for a, b in zip(nodes, nodes[1:]):
        draw.line([a[0] + 28, a[1], b[0] - 28, b[1]], fill=ink, width=2)
    draw.line([250, y + 70, nodes[0][0] - 28, nodes[0][1]], fill=ink, width=2)

    draw.text((80, height - 70), "Pipeline: photo → warp → cleanup → OCR → structure → export", font=small_f, fill=(70, 90, 78))

    return img


def add_perspective_and_glare(flat: Image.Image) -> Image.Image:
    """Skew the board and add glare so before/after demos look real."""
    arr = cv2.cvtColor(np.array(flat), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]

    # Destination canvas larger (room background)
    canvas_h, canvas_w = int(h * 1.15), int(w * 1.2)
    canvas = np.full((canvas_h, canvas_w, 3), (48, 52, 48), dtype=np.uint8)
    # Wall texture
    noise = np.random.default_rng(7).integers(0, 18, (canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas = cv2.add(canvas, noise)

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    # Slight trapezoid (phone angle)
    margin_x, margin_y = 70, 55
    dst = np.float32(
        [
            [margin_x + 40, margin_y + 10],
            [canvas_w - margin_x - 10, margin_y + 55],
            [canvas_w - margin_x - 55, canvas_h - margin_y - 20],
            [margin_x + 10, canvas_h - margin_y - 70],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (canvas_w, canvas_h), borderValue=(48, 52, 48))

    # Blend onto canvas where warped is non-border
    mask = (warped.sum(axis=2) > 10).astype(np.uint8) * 255
    mask3 = cv2.merge([mask, mask, mask])
    scene = np.where(mask3 > 0, warped, canvas).astype(np.uint8)

    # Specular glare blob
    glare = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    cx, cy = int(canvas_w * 0.62), int(canvas_h * 0.28)
    for y in range(canvas_h):
        for x in range(max(0, cx - 180), min(canvas_w, cx + 180)):
            d = math.hypot(x - cx, y - cy) / 140.0
            if d < 1.4:
                glare[y, x] = max(glare[y, x], math.exp(-d * d * 2.2) * 70)
    scene = np.clip(scene.astype(np.float32) + glare[:, :, None], 0, 255).astype(np.uint8)

    # Mild vignette / phone-cam softness
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    return Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    flat = draw_flat_board()
    angled = add_perspective_and_glare(flat)

    flat_path = SAMPLES / "whiteboard_flat.jpg"
    sample_path = SAMPLES / "whiteboard_sample.jpg"
    flat.convert("RGB").save(flat_path, quality=92)
    angled.convert("RGB").save(sample_path, quality=90)
    print(f"Wrote {flat_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":
    main()
