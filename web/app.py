from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from boardnotes.pipeline import convert

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "samples"
ARTIFACTS = Path(tempfile.gettempdir()) / "boardnotes-artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BoardNotes", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": "BoardNotes"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "product": "boardnotes",
        "ocr_backend": os.environ.get("BOARDNOTES_OCR_BACKEND", "auto"),
        "lightweight": os.environ.get("BOARDNOTES_LIGHTWEIGHT", "0"),
    }


@app.get("/api/artifact/{token}/{kind}")
async def get_artifact(token: str, kind: str) -> FileResponse:
    allowed = {
        "original": "original.jpg",
        "rectified": "rectified.jpg",
        "cleaned": "cleaned.jpg",
        "markdown": "notes.md",
        "pdf": "notes.pdf",
    }
    if kind not in allowed and not kind.startswith("diagram_"):
        return JSONResponse({"error": "Unknown artifact"}, status_code=404)  # type: ignore[return-value]

    base = ARTIFACTS / token
    if kind.startswith("diagram_"):
        path = base / "diagrams" / f"{kind}.png"
        if not path.exists():
            # try diagram_0 style already
            path = base / "diagrams" / f"{kind}.png"
        if not path.exists():
            return JSONResponse({"error": "Not found"}, status_code=404)  # type: ignore[return-value]
        return FileResponse(path, media_type="image/png")

    path = base / allowed[kind]
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)  # type: ignore[return-value]
    media = {
        "original": "image/jpeg",
        "rectified": "image/jpeg",
        "cleaned": "image/jpeg",
        "markdown": "text/markdown; charset=utf-8",
        "pdf": "application/pdf",
    }[kind]
    filename = allowed[kind]
    return FileResponse(path, media_type=media, filename=filename)


@app.post("/api/convert")
async def api_convert(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
) -> JSONResponse:
    tmp_path: Path | None = None
    try:
        use_fixture = False
        if sample in {"whiteboard", "sample", "1"}:
            path = SAMPLES / "whiteboard_sample.jpg"
            if not path.exists():
                return JSONResponse(
                    {"error": "Sample missing. Run: python scripts/generate_samples.py"},
                    status_code=404,
                )
            file_name = path.name
            use_fixture = True
        elif file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower() or ".jpg"
            allowed = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
            if suffix not in allowed:
                return JSONResponse(
                    {"error": "Upload a photo (JPEG/PNG/WebP/TIFF/BMP)."},
                    status_code=400,
                )
            raw = await file.read()
            if len(raw) > 20 * 1024 * 1024:
                return JSONResponse({"error": "File too large (20 MB max)."}, status_code=400)
            fd, name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp_path = Path(name)
            tmp_path.write_bytes(raw)
            path = tmp_path
            file_name = file.filename
        else:
            return JSONResponse({"error": "Upload a photo or try the sample."}, status_code=400)

        token = uuid.uuid4().hex
        out_dir = ARTIFACTS / token
        # Free-tier / CI friendly defaults
        ocr_backend = os.environ.get("BOARDNOTES_OCR_BACKEND") or None
        result = convert(
            path,
            out_dir=out_dir,
            ocr_backend=ocr_backend,
            pdf=True,
            use_fixture=use_fixture if use_fixture else None,
            file_name=file_name,
        )

        diagram_urls = []
        for i, d in enumerate(result.diagrams):
            diagram_urls.append(
                {
                    "url": f"/api/artifact/{token}/diagram_{i}",
                    "label": d.label,
                }
            )

        payload = {
            **result.to_dict(),
            "token": token,
            "original_url": f"/api/artifact/{token}/original",
            "rectified_url": f"/api/artifact/{token}/rectified",
            "cleaned_url": f"/api/artifact/{token}/cleaned",
            "markdown_url": f"/api/artifact/{token}/markdown",
            "pdf_url": f"/api/artifact/{token}/pdf",
            "diagram_urls": diagram_urls,
            "markdown": result.markdown,
        }
        return JSONResponse(payload)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
