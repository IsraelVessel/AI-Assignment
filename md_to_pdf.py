"""Simple markdown to PDF converter using reportlab for basic formatting.
Run: python md_to_pdf.py
"""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

import textwrap

INPUT = 'AI_Tools_Assignment_Report.md'
OUTPUT = 'AI_Tools_Assignment_Report_with_images.pdf'

def md_to_text_lines(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Simple conversion: keep lines but collapse multiple newlines
    out = []
    for ln in lines:
        out.append(ln.rstrip('\n'))
    return out


def create_pdf(lines, out_path):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 20*mm
    y = height - margin
    max_width = width - 2*margin
    wrap_cols = 100
    for line in lines:
        if y < margin:
            c.showPage()
            y = height - margin
        if line.strip() == '':
            y -= 6
            continue
        # headings
        if line.startswith('#'):
            txt = line.lstrip('#').strip()
            c.setFont('Helvetica-Bold', 14)
            wrapped = textwrap.wrap(txt, 90)
            for w in wrapped:
                c.drawString(margin, y, w)
                y -= 16
            y -= 6
            c.setFont('Helvetica', 10)
        else:
            wrapped = textwrap.wrap(line, 100)
            for w in wrapped:
                c.drawString(margin, y, w)
                y -= 12
    c.save()


def add_images(canvas_path, out_path):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 20*mm
    y = height - margin
    # First, draw the existing text PDF as background by creating a new page and then adding images
    # Simpler approach: write text first using create_pdf into out_path, then reopen and append images.
    # For simplicity, just add images on new pages after the text.
    create_pdf(md_to_text_lines(INPUT), out_path)
    # Append images
    from reportlab.lib.utils import ImageReader
    img_files = ['screenshots/iris_report.png', 'screenshots/spacy_ner.png', 'screenshots/mnist_samples.png']
    for img in img_files:
        if os.path.exists(img):
            c.drawImage(ImageReader(img), margin, height/2 - 50*mm, width=width-2*margin, preserveAspectRatio=True, anchor='c')
            c.showPage()
    c.save()


if __name__ == '__main__':
    lines = md_to_text_lines(INPUT)
    create_pdf(lines, OUTPUT)
    print('PDF written to', OUTPUT)
