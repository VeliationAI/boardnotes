"""Export structured notes as Markdown and PDF."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from boardnotes.models import DiagramRegion


def write_markdown(markdown: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _md_to_flowables(markdown: str, base_dir: Path, styles: dict):
    flow = []
    lines = markdown.splitlines()
    i = 0
    bullet_buf: list[str] = []
    numbered_buf: list[str] = []

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = [ListItem(Paragraph(_escape(t), styles["Body"])) for t in bullet_buf]
        flow.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
        flow.append(Spacer(1, 6))
        bullet_buf = []

    def flush_numbered():
        nonlocal numbered_buf
        if not numbered_buf:
            return
        items = [ListItem(Paragraph(_escape(t), styles["Body"])) for t in numbered_buf]
        flow.append(ListFlowable(items, bulletType="1", leftIndent=18))
        flow.append(Spacer(1, 6))
        numbered_buf = []

    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line:
            flush_bullets()
            flush_numbered()
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if line.startswith("#"):
            flush_bullets()
            flush_numbered()
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            style = styles["H1"] if level <= 1 else styles["H2"] if level == 2 else styles["H3"]
            flow.append(Paragraph(_escape(text), style))
            flow.append(Spacer(1, 8))
            i += 1
            continue

        m_img = img_re.search(line)
        if m_img:
            flush_bullets()
            flush_numbered()
            rel = m_img.group(2)
            img_path = Path(rel)
            if not img_path.is_file():
                img_path = base_dir / rel
            if img_path.is_file():
                try:
                    flow.append(RLImage(str(img_path), width=5.5 * inch, height=2.2 * inch, kind="proportional"))
                    flow.append(Spacer(1, 8))
                except Exception:  # noqa: BLE001
                    flow.append(Paragraph(f"[diagram: {rel}]", styles["Body"]))
            else:
                flow.append(Paragraph(f"[diagram: {rel}]", styles["Body"]))
            i += 1
            continue

        if line.startswith(("- ", "* ", "• ")):
            flush_numbered()
            bullet_buf.append(line[2:].strip() if line[1] == " " else line[1:].strip())
            i += 1
            continue

        m_num = re.match(r"^(\d+)[.)]\s+(.*)$", line)
        if m_num:
            flush_bullets()
            numbered_buf.append(m_num.group(2))
            i += 1
            continue

        if line.startswith(">"):
            flush_bullets()
            flush_numbered()
            flow.append(Paragraph(_escape(line.lstrip("> ").strip()), styles["Quote"]))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        flush_bullets()
        flush_numbered()
        flow.append(Paragraph(_escape(line), styles["Body"]))
        flow.append(Spacer(1, 4))
        i += 1

    flush_bullets()
    flush_numbered()
    return flow


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_pdf(
    markdown: str,
    path: Path,
    diagrams: list[DiagramRegion] | None = None,
    title: str = "BoardNotes",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = path.parent

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
        author="BoardNotes by Veliation AI",
    )
    styles = getSampleStyleSheet()
    custom = {
        "H1": ParagraphStyle(
            "BN_H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            spaceAfter=6,
            textColor="#1a2e24",
        ),
        "H2": ParagraphStyle(
            "BN_H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            spaceBefore=8,
            textColor="#2c4a3a",
        ),
        "H3": ParagraphStyle(
            "BN_H3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor="#3a5a48",
        ),
        "Body": ParagraphStyle(
            "BN_Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor="#1f2a24",
        ),
        "Quote": ParagraphStyle(
            "BN_Quote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leftIndent=12,
            textColor="#3d5248",
            backColor="#eef3ef",
        ),
    }

    story = _md_to_flowables(markdown, base_dir, custom)
    if not story:
        story = [Paragraph("(empty notes)", custom["Body"])]
    doc.build(story)
    return path


def write_exports(
    markdown: str,
    out_dir: Path,
    stem: str = "notes",
    diagrams: list[DiagramRegion] | None = None,
    pdf: bool = True,
) -> tuple[Path, Path | None]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = write_markdown(markdown, out_dir / f"{stem}.md")
    pdf_path = None
    if pdf:
        pdf_path = write_pdf(markdown, out_dir / f"{stem}.pdf", diagrams=diagrams)
    return md_path, pdf_path
