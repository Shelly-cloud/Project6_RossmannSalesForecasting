"""Renders reports/blog_post.md to reports/Rossmann_Sales_Forecasting_Blog.pdf.

Minimal hand-rolled markdown-to-PDF using fpdf2 -- no external binary
dependency (weasyprint/LibreOffice aren't available in this environment).
Handles headings, bold/italic inline spans, tables, bullet lists and plain
paragraphs, which covers everything blog_post.md actually uses.
"""
import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "blog_post.md"
OUT = ROOT / "reports" / "Rossmann_Sales_Forecasting_Blog.pdf"

INK = (11, 11, 11)
SECONDARY = (82, 81, 78)
BLUE = (42, 120, 214)
MUTED = (137, 135, 129)
GRIDLINE = (225, 224, 217)


def clean_inline(text: str) -> str:
    """Strip markdown emphasis/code markers and normalise unicode punctuation
    fpdf2's core (latin-1) fonts can't render, keeping the plain text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    replacements = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b2": "^2",
        "\u2192": "->", "\u00d7": "x", "\u2264": "<=", "\u2265": ">=",
        "\u00b1": "+/-", "\u2212": "-", "\u207b": "^-", "\u2075": "5",
        "\u2074": "4", "\u00b3": "^3", "\u00b9": "^1",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


class BlogPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def render(md_text: str) -> BlogPDF:
    pdf = BlogPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_text_color(*INK)

    def merge_paragraphs(raw_lines: list[str]) -> list[str]:
        """Join hard-wrapped source lines back into one logical line per block.

        blog_post.md hard-wraps prose (and list items) at ~100 columns, so a
        bold/italic span can open on one source line and close on the next --
        a plain per-line regex never sees both halves together, and a list
        item's continuation text would otherwise be mistaken for its own
        paragraph. A new block starts at a heading, table row, rule, blank
        line, or a fresh list item; anything else continues the block
        currently open. multi_cell re-wraps the merged text to the PDF's own
        column width regardless of the original line breaks.
        """
        starts_block = lambda ln: (
            ln.startswith(("#", "|", "---")) or re.match(r"^(-|\*)\s", ln) or re.match(r"^\d+\.\s", ln)
        )
        merged: list[str] = []
        current: str | None = None

        def flush():
            nonlocal current
            if current is not None:
                merged.append(current)
                current = None

        for ln in raw_lines:
            if not ln.strip():
                flush()
                merged.append("")
            elif starts_block(ln):
                flush()
                current = ln
            elif current is not None:
                current += " " + ln.strip()
            else:
                current = ln.strip()
        flush()
        return merged

    lines = merge_paragraphs(md_text.split("\n"))
    i = 0
    in_table = False
    table_rows: list[list[str]] = []

    def wrap_line_count(text: str, width: float) -> int:
        """Simulate fpdf2's word-wrap to count how many lines a cell needs.

        multi_cell's own wrapping is only known once it renders, but a row's
        height must be fixed *before* rendering any cell in it (cells are
        drawn side by side, so the row can't grow while a sibling cell is
        already on the page). Pre-computing the wrap with the same
        greedy-fill algorithm keeps the row heights consistent with what
        multi_cell will actually draw.
        """
        words = text.split(" ")
        if not words:
            return 1
        n_lines = 1
        cur = words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if pdf.get_string_width(trial) <= width - 2:  # ~1mm padding each side
                cur = trial
            else:
                n_lines += 1
                cur = w
        return n_lines

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        header, *body = table_rows
        n = len(header)
        page_w = pdf.w - pdf.r_margin - pdf.l_margin
        col_w = page_w / n
        left = pdf.l_margin

        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(*GRIDLINE)
        header_lines = max(wrap_line_count(clean_inline(c), col_w) for c in header)
        header_h = 6 * header_lines
        pdf.set_x(left)
        for c in header:
            pdf.multi_cell(col_w, 6, clean_inline(c), border=0, align="L", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_xy(left, pdf.get_y() + header_h)

        pdf.set_font("Helvetica", "", 8.5)
        for row in body:
            max_lines = max(wrap_line_count(clean_inline(c), col_w) for c in row)
            row_h = 5.5 * max_lines
            x0, y0 = left, pdf.get_y()
            pdf.set_x(x0)
            for c in row:
                pdf.multi_cell(col_w, 5.5, clean_inline(c), border="B", align="L", new_x="RIGHT", new_y="TOP")
            pdf.set_xy(x0, y0 + row_h)
        pdf.ln(4)
        pdf.set_x(left)
        table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                table_rows.append(cells)
            i += 1
            continue
        else:
            flush_table()

        def full_width(h, text, indent=0):
            # fpdf2's multi_cell does not reliably reset x to the left margin
            # between calls in this version, so every full-width block resets
            # it explicitly before drawing rather than trusting the default.
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(pdf.w - pdf.r_margin - pdf.l_margin - indent, h, text,
                            new_x="LMARGIN", new_y="NEXT")

        if not line:
            pdf.ln(2)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(*INK)
            full_width(10, clean_inline(line[2:]))
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(*BLUE)
            full_width(8, clean_inline(line[3:]))
            pdf.set_text_color(*INK)
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            full_width(7, clean_inline(line[4:]))
            pdf.ln(1)
        elif line.startswith("---"):
            pdf.set_draw_color(*GRIDLINE)
            y = pdf.get_y() + 1
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            pdf.set_font("Helvetica", "I", 9.5)
            pdf.set_text_color(*MUTED)
            full_width(5.5, clean_inline(line.strip("*")))
            pdf.set_text_color(*INK)
            pdf.ln(1)
        elif re.match(r"^\d+\.\s", line):
            pdf.set_font("Helvetica", "", 10.5)
            text = re.sub(r"^\d+\.\s", "", line)
            num = re.match(r"^(\d+)\.", line).group(1)
            full_width(6, f"{num}. {clean_inline(text)}", indent=4)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 10.5)
            full_width(6, f"-  {clean_inline(line[2:])}", indent=4)
        else:
            pdf.set_font("Helvetica", "", 10.5)
            full_width(6, clean_inline(line))
        i += 1

    flush_table()
    return pdf


md_text = SRC.read_text(encoding="utf-8")
pdf = render(md_text)
pdf.output(str(OUT))
print("wrote", OUT)
