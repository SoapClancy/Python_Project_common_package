import docx
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT as WD_ALIGN_PARAGRAPH


def new_document():
    DOCUMENT = Document()

    style = DOCUMENT.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)

    obj_styles = DOCUMENT.styles
    obj_styles.add_style('Table', WD_STYLE_TYPE.PARAGRAPH)
    style = DOCUMENT.styles['Table']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    paragraph_format = style.paragraph_format
    paragraph_format.space_before = Pt(0)
    paragraph_format.space_after = Pt(0)
    return DOCUMENT
