import os
import winreg
import re
from typing import Iterable, Tuple
from functools import singledispatch
from pathlib import Path


def try_to_find_file(file_path):
    """
    寻找结果文件。有的话就返回True，如果没有的话则返回False
    """
    if file_path is None:
        return None
    return os.path.isfile(file_path)


def try_to_find_file_if_exist_then_delete(file_):
    if try_to_find_file(file_):
        os.remove(file_)


# 使用泛型函数
@singledispatch
def try_to_find_folder_path_otherwise_make_one(folder_path):
    assert (isinstance(folder_path, tuple)) or isinstance(folder_path, str) or isinstance(folder_path, Path)


@try_to_find_folder_path_otherwise_make_one.register(str)
def _(folder_path: str):
    if os.path.exists(folder_path):
        return True
    else:
        os.makedirs(folder_path)
        return False


@try_to_find_folder_path_otherwise_make_one.register(tuple)
def _(folder_path: Tuple[str, ...]):
    for i in folder_path:
        try_to_find_folder_path_otherwise_make_one(i)


@try_to_find_folder_path_otherwise_make_one.register(Path)
def _(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)


def list_all_specific_format_files_in_a_folder_path(folder_path: str, format_: str, order: str = 'time'):
    files = os.listdir(folder_path)
    files = [x for x in files if re.search(r'\.' + format_ + '$', x)]
    files = [folder_path + x for x in files]
    if order == 'time':
        files = sorted(files, key=lambda x: os.path.getctime(x))
    return files


def remove_win10_max_path_limit():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem",
                         0, winreg.KEY_WRITE)
    winreg.SetValueEx(key, r"LongPathsEnabled", 0, winreg.REG_DWORD, 1)


def restore_win10_max_path_limit():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem",
                         0, winreg.KEY_WRITE)
    winreg.SetValueEx(key, r"LongPathsEnabled", 0, winreg.REG_DWORD, 0)
