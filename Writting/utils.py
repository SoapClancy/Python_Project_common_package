from typing import List, Union
from pathlib import Path
from Writting import docx_document_template_to_collect_figures
from docx import Document
from docx.shared import Cm, Pt
from File_Management.path_and_file_management_Func import try_to_find_file_if_exist_then_delete, \
    list_all_specific_format_files_in_a_path


def put_png_file_into_a_docx(png_file_in_a_dict: dict, docx_file_path: Union[str, Path], cols: int = 2):
    """
    png_file_in_a_dict： key就是名字，value的形式是[图像，宽度]
    """
    document = docx_document_template_to_collect_figures()
    if cols == 2:
        rows = int(png_file_in_a_dict.__len__()) + 1
    elif cols == 1:
        rows = 2 * png_file_in_a_dict.__len__() + 1
    else:
        raise Exception("cols should be 1 or 2")
    table = document.add_table(rows=rows, cols=cols)
    for i, (key, val) in enumerate(png_file_in_a_dict.items()):
        if val[0] is None:
            continue
        if cols == 2:
            row = int(i / 2) * 2
            col = i % 2
            this_cell = table.rows[row].cells[col]
            this_cell.paragraphs[0].add_run().add_picture(val[0], width=Cm(7.5))

            table.rows[row + 1].cells[col].paragraphs[0].add_run(key)
        else:
            row = i * 2
            col = 0
            this_cell = table.rows[row].cells[col]
            this_cell.paragraphs[0].add_run().add_picture(val[0], width=Cm(15))

            table.rows[row + 1].cells[col].paragraphs[0].add_run(key)

    try_to_find_file_if_exist_then_delete(docx_file_path)
    document.save(docx_file_path)


def put_all_png_in_a_path_into_a_docx(path_: Union[str, Path], docx_file_path: Union[str, Path]):
    files = list_all_specific_format_files_in_a_path(path_, 'png', '')
    to_do_ = {f'{i}': (x,) for i, x in enumerate(files)}
    put_png_file_into_a_docx(to_do_, docx_file_path)


def put_cached_png_into_a_docx(cached_png: dict,
                               docx_file_path: Union[str, Path],
                               cols: int):
    """
    :param cached_png: 一个dict，key代表标题，value代表图像（BytesIO()和宽度）
    :param docx_file_path
    :param cols
    """
    put_png_file_into_a_docx(cached_png, docx_file_path, cols)


def put_cached_png_into_a_docx_new():
    pass
