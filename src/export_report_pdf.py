from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
INPUT_PATH = REPORTS_DIR / "informe_miniproyecto_2.md"
OUTPUT_PATH = REPORTS_DIR / "informe_miniproyecto_2.pdf"
COURSE_NAME = "Fundamentos y Aplicaciones de Inteligencia Artificial"
UNIVERSITY_NAME = "Universidad Autonoma de Occidente"
PROFESSOR_NAME = "Juan Sebastian Mosquera Maturana"
PROJECT_NAME = "MiniProyecto 2 - Clasificacion de niveles de obesidad"
AUTHORS = [
    "Valentina Popo Montilla",
    "Juan Camilo Balleresteros Sierra",
    "Santigo Rodriguez Gacha",
]


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenterTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenterSubTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenterBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles["Heading1"].spaceBefore = 14
    styles["Heading1"].spaceAfter = 10
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 8
    return styles


def format_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def image_flowable(image_markdown: str):
    match = re.match(r"!\[(.*?)\]\((.*?)\)", image_markdown.strip())
    if not match:
        return []
    alt_text, image_path = match.groups()
    full_path = REPORTS_DIR / image_path
    if not full_path.exists():
        return []
    img = Image(str(full_path))
    img._restrictSize(17 * cm, 18 * cm)
    caption = Paragraph(f"<i>{format_inline(alt_text)}</i>", build_styles()["CenterBody"])
    return [img, Spacer(1, 0.15 * cm), caption, Spacer(1, 0.35 * cm)]


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped.replace("|", "").replace(" ", "")) == {"-"}:
            continue
        rows.append([format_inline(cell.strip()) for cell in stripped.strip("|").split("|")])
    return rows


def add_cover(story: list, styles) -> None:
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph(UNIVERSITY_NAME, styles["CenterSubTitle"]))
    story.append(Paragraph(COURSE_NAME, styles["CenterSubTitle"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(PROJECT_NAME, styles["CenterTitle"]))
    story.append(Paragraph("Informe final", styles["CenterBody"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(f"Profesor: {PROFESSOR_NAME}", styles["CenterBody"]))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles["CenterBody"]))
    story.append(Spacer(1, 1.1 * cm))
    story.append(Paragraph("<b>Integrantes</b>", styles["CenterBody"]))
    for author in AUTHORS:
        story.append(Paragraph(author, styles["CenterBody"]))
    story.append(PageBreak())


def build_story(markdown_text: str):
    styles = build_styles()
    story = []
    add_cover(story, styles)

    lines = markdown_text.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(
                    Paragraph(
                        "<br/>".join(format_inline(code) for code in code_lines),
                        styles["Code"],
                    )
                )
                story.append(Spacer(1, 0.2 * cm))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(format_inline(stripped[2:].strip()), styles["Title"]))
            story.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(format_inline(stripped[3:].strip()), styles["Heading1"]))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(format_inline(stripped), styles["BodyJustify"]))
            i += 1
            continue

        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if content.startswith("!["):
                story.extend(image_flowable(content))
            else:
                story.append(Paragraph(format_inline(f"• {content}"), styles["BodyJustify"]))
            i += 1
            continue

        if stripped.startswith("!["):
            story.extend(image_flowable(stripped))
            i += 1
            continue

        if "|" in stripped:
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                table = Table(rows, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7f5")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fbff")]),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.25 * cm))
            continue

        story.append(Paragraph(format_inline(stripped), styles["BodyJustify"]))
        i += 1

    return story


def main() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    story = build_story(INPUT_PATH.read_text(encoding="utf-8"))
    document.build(story)
    print(f"Reporte PDF generado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
