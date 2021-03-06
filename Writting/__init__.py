import docx
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT as WD_ALIGN_PARAGRAPH


def docx_document_template_to_collect_figures():
    this_document = Document()

    style = this_document.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)

    obj_styles = this_document.styles
    obj_styles.add_style('Table', WD_STYLE_TYPE.PARAGRAPH)
    style = this_document.styles['Table']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    paragraph_format = style.paragraph_format
    paragraph_format.space_before = Pt(0)
    paragraph_format.space_after = Pt(0)
    return this_document


