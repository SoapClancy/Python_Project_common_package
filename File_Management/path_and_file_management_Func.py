import os
import winreg
import re
from typing import Iterable, Tuple
from functools import singledispatch


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
def try_to_find_path_otherwise_make_one(path_):
    assert (isinstance(path_, tuple)) or isinstance(path_, str)


@try_to_find_path_otherwise_make_one.register(str)
def _(path_: str):
    if os.path.exists(path_):
        return True
    else:
        os.makedirs(path_)
        return False


@try_to_find_path_otherwise_make_one.register(tuple)
def _(path_: Tuple[str, ...]):
    for i in path_:
        try_to_find_path_otherwise_make_one(i)


def list_all_specific_format_files_in_a_path(path_: str, format_: str, order: str = 'time'):
    files = os.listdir(path_)
    files = [x for x in files if re.search('\.' + format_ + '$', x)]
    files = [path_ + x for x in files]
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
