from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from boardnotes.pipeline import convert

app = typer.Typer(
    name="boardnotes",
    help="BoardNotes — whiteboard photo → clean structured digital notes.",
    add_completion=False,
)
console = Console()


@app.command("convert")
def convert_cmd(
    photo: Path = typer.Argument(..., exists=True, readable=True, help="Whiteboard photo"),
    out: Path = typer.Option(Path("./out"), "--out", "-o", help="Output directory"),
    ocr: Optional[str] = typer.Option(
        None,
        "--ocr",
        help="OCR backend: auto|trocr|easyocr|tesseract|mock",
    ),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF export"),
    print_md: bool = typer.Option(False, "--print", help="Print Markdown to terminal"),
) -> None:
    """Convert a whiteboard photo into Markdown (+ PDF) notes."""
    result = convert(
        photo,
        out_dir=out,
        ocr_backend=ocr,
        pdf=not no_pdf,
    )

    console.print(
        Panel.fit(
            f"[bold]{result.file_name}[/bold]\n"
            f"{result.width}×{result.height} · OCR={result.ocr_backend} · "
            f"warped={'yes' if result.warped else 'no'}\n"
            f"{len(result.lines)} lines · {len(result.diagrams)} diagrams",
            title="BoardNotes",
            border_style="green",
        )
    )
    for note in result.notes:
        console.print(f"[dim]• {note}[/dim]")

    if result.markdown_path:
        console.print(f"[green]Markdown[/green] {result.markdown_path}")
    if result.pdf_path:
        console.print(f"[green]PDF[/green]      {result.pdf_path}")
    if result.rectified_path:
        console.print(f"[green]Rectified[/green] {result.rectified_path}")
    if result.cleaned_path:
        console.print(f"[green]Cleaned[/green]   {result.cleaned_path}")

    if print_md:
        console.print()
        console.print(Markdown(result.markdown))


@app.command("version")
def version_cmd() -> None:
    from boardnotes import __version__

    console.print(f"BoardNotes {__version__}")


if __name__ == "__main__":
    app()
