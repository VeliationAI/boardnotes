# BoardNotes

**Whiteboard photo → clean structured digital notes.**

Upload or snap a whiteboard. BoardNotes corrects perspective, cleans glare/contrast, runs handwriting OCR, detects structure (headings / bullets / numbering), keeps diagram crops, and exports searchable **Markdown + PDF**.

Built by [Veliation AI](https://github.com/VeliationAI) · MIT.

## Pipeline

1. Perspective correction (find corners → warp flat)
2. Cleanup (glare / CLAHE contrast / denoise / sharpen)
3. OCR (TrOCR → EasyOCR → Tesseract → mock/demo fixture)
4. Structure → Markdown hierarchy
5. Diagram regions cropped and embedded
6. Export Markdown + optional PDF

## Honest v0.1 limits

- **One photo at a time** — no multi-photo stitch
- **English first** — Telugu/Hindi later
- **Diagrams kept as cropped images**, not redrawn as vectors

## OCR & free-tier hosting

Heavy models (TrOCR / torch, EasyOCR) are **optional**. The core OpenCV pipeline always runs.

| Mode | How | Best for |
|---|---|---|
| `mock` (default on Render free) | Deterministic demo fixture for the synthetic sample; graceful placeholder otherwise | CI, free Render |
| `tesseract` | `pip install pytesseract` + system Tesseract | Light CPU OCR |
| `easyocr` | `pip install easyocr` | Better handwriting, heavier |
| `trocr` | `pip install transformers torch` | Best handwriting when you have disk/RAM |
| `auto` | Prefer TrOCR → EasyOCR → Tesseract → mock | Local installs |

Environment flags:

```bash
export BOARDNOTES_LIGHTWEIGHT=1          # skip TrOCR/EasyOCR
export BOARDNOTES_OCR_BACKEND=mock       # force demo path
# or: auto | trocr | easyocr | tesseract | mock
```

On Render free tier, the app ships with `BOARDNOTES_LIGHTWEIGHT=1` and `BOARDNOTES_OCR_BACKEND=mock` so deploys stay small. The **Try sample board** button still produces impressive structured Markdown via the built-in fixture that matches the synthetic whiteboard.

## Quick start (CLI)

```bash
cd boardnotes
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_samples.py
boardnotes convert samples/whiteboard_sample.jpg --out ./out --ocr mock
```

## Web app

```bash
pip install -e .
uvicorn web.app:app --reload --port 8000
```

Open http://localhost:8000 — drag-drop a photo or tap **Try sample board**. Compare original vs rectified, then download MD/PDF.

## Demo before/after (LinkedIn-ready)

| File | Role |
|---|---|
| `samples/whiteboard_sample.jpg` | Angled phone shot + glare (before) |
| `out/rectified.jpg` / UI “Rectified” pane | Flattened board (after) |
| `out/notes.md` | Structured searchable notes |

Regenerate synthetic boards (no real classroom photos):

```bash
python scripts/generate_samples.py
```

## Tests

```bash
pytest -q
```

OCR is mocked in unit tests; perspective, cleanup, and structure heuristics are covered for real.

## Docker

```bash
docker build -t boardnotes .
docker run -p 8000:8000 -e PORT=8000 boardnotes
```

## License

MIT © Veliation AI / BoardNotes contributors
