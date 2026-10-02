"""Build data/sample/customers_sample.pdf, a multi-page customer table for trying PDF upload.

Usage:  python scripts/make_sample_pdf.py [--rows 60]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import ROOT
from src.data import load_raw

OUT = ROOT / "data" / "sample" / "customers_sample.pdf"


def main(n_rows: int) -> None:
    df = load_raw().drop(columns=["Churn"]).sample(n_rows, random_state=7)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=6.5, leading=8)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    data = [[Paragraph(c, head) for c in df.columns]]
    data += [[Paragraph(str(v), cell) for v in row] for row in df.itertuples(index=False)]

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C1D95")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F0FF")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C4B5FD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A3), leftMargin=20, rightMargin=20,
                            topMargin=24, bottomMargin=24, title="Customer list")
    title = Paragraph("Customer list: churn review", ParagraphStyle("t", fontSize=14, leading=18))
    doc.build([title, Spacer(1, 8), table])
    print(f"Wrote {OUT} ({n_rows} customers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60)
    main(parser.parse_args().rows)
