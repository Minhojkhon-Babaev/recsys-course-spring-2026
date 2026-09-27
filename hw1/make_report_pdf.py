#!/usr/bin/env python3
"""Одностраничный PDF-отчёт для All Cups."""

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "description.pdf"

FONT_REG = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")


def main() -> None:
    pdfmetrics.registerFont(TTFont("Report", str(FONT_REG)))
    pdfmetrics.registerFont(TTFont("Report-Bold", str(FONT_BOLD if FONT_BOLD.exists() else FONT_REG)))

    title = ParagraphStyle(
        "title",
        fontName="Report-Bold",
        fontSize=16,
        leading=20,
        spaceAfter=4,
        alignment=TA_LEFT,
    )
    subtitle = ParagraphStyle(
        "subtitle",
        fontName="Report",
        fontSize=11,
        leading=14,
        spaceAfter=10,
    )
    heading = ParagraphStyle(
        "heading",
        fontName="Report-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "body",
        fontName="Report",
        fontSize=12,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    result = ParagraphStyle(
        "result",
        fontName="Report",
        fontSize=12,
        leading=16,
        leftIndent=8,
        spaceAfter=2,
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="ДЗ1. Описание решения",
    )
    story = [
        Paragraph("ДЗ1. Collaborative filtering рекомендации музыки", title),
        Paragraph("Описание решения", subtitle),
        Paragraph(
            "Задача — для каждой пары пользователь–трек из теста поставить скор, "
            "по которому треки пользователя выстроятся в правильном порядке. "
            "Релевантность в обучении — доля дослушивания time.",
            body,
        ),
        Paragraph(
            "Сначала по истории прослушиваний строю матрицу пользователь–трек. "
            "На ней учу две коллаборативные модели: SVD и ALS. ALS берёт time "
            "как уверенность в факте прослушивания. Для каждой пары дополнительно "
            "считаю простые признаки: насколько этот пользователь обычно дослушивает "
            "треки, насколько трек вообще дослушивают, сколько у них событий.",
            body,
        ),
        Paragraph(
            "Эти признаки и скоры SVD/ALS склеиваю и отдаю в линейную регрессию Ridge. "
            "Она предсказывает долю дослушивания. Чтобы не подогнаться под те же пары, "
            "на которых учились SVD и ALS, Ridge настраиваю на отложенных событиях "
            "случайных пользователей. В тесте скор пары — это как раз предсказание Ridge.",
            body,
        ),
        Paragraph(
            "Данные симулятора и любые внешние источники не использовал.",
            body,
        ),
        Paragraph("Результаты All Cups", heading),
        Paragraph("public NDCG&nbsp;&nbsp;0.954994", result),
        Paragraph("private NDCG&nbsp;&nbsp;0.954777", result),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
