"""PDF fixture builder for parser tests.

Builds minimal, deterministic PDFs in-memory with reportlab so tests never touch
the network or the real corpus. Layout is fixed so assertions on block order,
page numbers and section paths are stable.
"""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

HEADER = "Microsoft 2023 Annual Report"


def _decorate(pdf: io.BytesIO, pages: list[list[str]]) -> None:
    c = canvas.Canvas(pdf, pagesize=letter)
    width, height = letter
    for index, lines in enumerate(pages, start=1):
        # Header and footer repeat on every page -> exercise the 60% rule.
        c.setFont("Helvetica", 8)
        c.drawString(0.75 * inch, height - 0.5 * inch, HEADER)
        c.drawCentredString(width / 2, 0.4 * inch, f"page {index}")
        c.setFont("Helvetica", 11)
        y = height - 1.0 * inch
        for line in lines:
            if line == "__TABLE__":
                y -= 0.3 * inch
                _draw_table(c, y)
                y -= 0.35 * inch * 4
                continue
            if line == "__PAGEBREAK__":
                break
            c.drawString(0.75 * inch, y, line)
            y -= 0.25 * inch
        c.showPage()
    c.save()


def _draw_table(c: canvas.Canvas, top: float) -> None:
    """Draw a ruled 3-column table so pdfplumber's 'lines' strategy finds it.

    Text baselines sit just *below* each horizontal rule. pdfplumber's line-based
    strategy binds characters to the cell whose rules enclose them, so a header
    drawn exactly on the top rule falls outside the detected grid and is silently
    lost — which is how a table ends up with its first data row promoted to the
    header row.
    """
    rows = [
        ("Metric", "FY2023", "FY2022"),
        ("Revenue", "$ 211,915", "$ 198,270"),
        ("Operating income", "$ 88,523", "$ 83,383"),
    ]
    x0, col_w, row_h = 0.75 * inch, 1.6 * inch, 0.22 * inch
    for r, row in enumerate(rows):
        y = top - (r + 1) * row_h + 0.05 * inch
        for cidx, cell in enumerate(row):
            c.drawString(x0 + cidx * col_w + 0.05 * inch, y, cell)
    # Rulings are what pdfplumber needs to detect cells.
    for r in range(len(rows) + 1):
        c.line(x0, top - r * row_h, x0 + 3 * col_w, top - r * row_h)
    for cidx in range(4):
        c.line(x0 + cidx * col_w, top, x0 + cidx * col_w, top - len(rows) * row_h)


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build PDF bytes from a list of pages, each a list of text lines."""
    buf = io.BytesIO()
    _decorate(buf, pages)
    return buf.getvalue()


def sample_doc() -> bytes:
    """A three-page document exercising every T2 invariant."""
    return build_pdf(
        [
            [
                "ANNUAL REPORT",
                "MANAGEMENT'S DISCUSSION",
                "Revenue grew across segments in fiscal 2023.",
                "The operat-",
                "ing margin expanded year over year.",
            ],
            [
                "SUMMARY OF RESULTS",
                "__TABLE__",
                "Note 13 discusses commitments in detail.",
            ],
            [
                "CONTINGENCIES",
                "We are subject to various claims.",
            ],
        ]
    )


def image_only_doc() -> bytes:
    """A page with no extractable text layer, for the text_layer: false path."""
    return build_pdf([["     ", "     "]])
