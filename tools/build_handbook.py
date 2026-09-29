#!/usr/bin/env python3
"""Compile the markdown design set into docs/handbook.pdf."""

from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "handbook.pdf"

NAVY = HexColor("#1b2430")
INK = HexColor("#1a1a1a")
MUTED = HexColor("#4a5560")
RULE = HexColor("#c5a46e")
CODE_BG = HexColor("#f4f1ea")

FILES = [
    "00-overview.md",
    "01-lineage.md",
    "02-rules-catalog.md",
    "03-mohr.md",
    "04-mace.md",
    "05-bins-integrate.md",
    "06-architecture.md",
    "07-determinism.md",
    "08-dgx-spark.md",
    "09-roadmap.md",
    "10-mapping-vita-entelechy.md",
    "11-config-schema.md",
    "12-ai-evaluator-deferred.md",
    "13-file-index.md",
    "14-shaders.md",
    "15-testing.md",
]


def styles():
    base = getSampleStyleSheet()
    s = {
        "cover": ParagraphStyle(
            "cover",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=26,
            leading=32,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "sub": ParagraphStyle(
            "sub",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=12,
            textColor=MUTED,
            spaceAfter=24,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=13,
            leading=17,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=15,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=INK,
            backColor=CODE_BG,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "th": ParagraphStyle(
            "th",
            fontName="Times-Bold",
            fontSize=8,
            leading=11,
            textColor=NAVY,
        ),
        "td": ParagraphStyle(
            "td",
            fontName="Times-Roman",
            fontSize=8,
            leading=11,
            textColor=INK,
        ),
        "li": ParagraphStyle(
            "li",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
            textColor=INK,
            leftIndent=8,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Times-Italic",
            fontSize=8,
            textColor=MUTED,
        ),
    }
    return s


def esc(t: str) -> str:
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline(t: str) -> str:
    t = esc(t)
    t = re.sub(r"`([^`]+)`", r"<font face='Courier' size='9'>\1</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    return t


def parse_table(lines: list[str], st) -> Table:
    rows = []
    for line in lines:
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cols)
    # drop separator
    body = [rows[0]] + [r for r in rows[1:] if not all(re.match(r"^:?-+:?$", c) for c in r)]
    data = []
    for i, row in enumerate(body):
        sty = st["th"] if i == 0 else st["td"]
        data.append([Paragraph(inline(c), sty) for c in row])
    tw = 7.0 * inch
    ncols = len(data[0])
    col_w = [tw / ncols] * ncols
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8eef4")),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#c8d0d8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def md_to_flow(text: str, st) -> list:
    flow = []
    lines = text.splitlines()
    i = 0
    in_code = False
    code = []
    para = []

    def flush_para():
        nonlocal para
        if para:
            flow.append(Paragraph(inline(" ".join(para)), st["body"]))
            para = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para()
            if in_code:
                block = "\n".join(code) if code else " "
                flow.append(Preformatted(block, st["code"], maxLineLength=92))
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            flush_para()
            tbl_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            flow.append(Spacer(1, 4))
            flow.append(parse_table(tbl_lines, st))
            flow.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            flush_para()
            flow.append(Paragraph(inline(line[2:]), st["h1"]))
        elif line.startswith("## "):
            flush_para()
            flow.append(Paragraph(inline(line[3:]), st["h2"]))
        elif line.startswith("- ") or line.startswith("* "):
            flush_para()
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
                items.append(ListItem(Paragraph(inline(lines[i][2:]), st["li"]), leftIndent=12))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", start="•"))
            continue
        elif re.match(r"^\d+\. ", line):
            flush_para()
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(Paragraph(inline(re.sub(r"^\d+\. ", "", lines[i])), st["li"]))
                i += 1
            for it in items:
                flow.append(it)
            continue
        elif line.strip() == "":
            flush_para()
        else:
            para.append(line.strip())
        i += 1
    flush_para()
    return flow


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, 10.55 * inch, 7.75 * inch, 10.55 * inch)
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 10.65 * inch, "Entelechy Continuum Engine — handbook")
    canvas.drawRightString(7.75 * inch, 10.65 * inch, "2026-09-28")
    canvas.line(0.75 * inch, 0.55 * inch, 7.75 * inch, 0.55 * inch)
    canvas.drawString(0.75 * inch, 0.38 * inch, "ArcSystems / ECE kernels 0.1.0")
    canvas.drawRightString(7.75 * inch, 0.38 * inch, f"{doc.page}")
    canvas.restoreState()


def main():
    st = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.7 * inch,
        title="ECE Handbook",
        author="ArcSystems",
    )
    story = [
        Paragraph("Entelechy Continuum Engine", st["cover"]),
        Paragraph("Corrected kernels, catalog, and build cut. Version 0.1.0.", st["sub"]),
        Paragraph(
            inline(
                "This handbook compiles the design documents in `docs/`. "
                "Executable truth is `ece/*.py` and `tests/`. The original "
                "catalog remains lineage reference; formulas here supersede it."
            ),
            st["body"],
        ),
        Spacer(1, 12),
    ]
    extra = [
        ROOT / "CORRECTIONS.md",
        ROOT / "CHANGELOG.md",
    ]
    for name in FILES:
        p = DOCS / name
        story.append(PageBreak())
        story.extend(md_to_flow(p.read_text(), st))
    for p in extra:
        story.append(PageBreak())
        story.extend(md_to_flow(p.read_text(), st))
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    main()
