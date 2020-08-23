import numpy as np
import pickle
from File_Management.path_and_file_management_Func import try_to_find_file, try_to_find_folder_path_otherwise_make_one
import re
from pathlib import Path
import functools


def load_npy_file(file_path: Path):
    """
    载入np文件。有的话就返回，如果没有的话则返回None
    """
    file_path = str(file_path)
    if try_to_find_file(file_path) is False:
        return None
    else:
        return np.load(file_path)


def save_npy_file(file_path: Path, array):
    """
    储存np文件。有的话就返回，如果没有的话则返回None
    """
    file_path = str(file_path)
    np.save(file_path, array)


def load_exist_npy_file_otherwise_run_and_save(file_path:Path):
    if not re.match(r".*\.npy$", str(file_path)):
        raise Exception("'file_path' should end with '.npy'")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(file_path, Path):
                try_to_find_folder_path_otherwise_make_one(file_path.parent)
            if load_npy_file(file_path) is not None:
                return load_npy_file(file_path)
            else:
                array = func(*args, **kwargs)
                save_npy_file(file_path, array)
                return array

        return wrapper

    return decorator


def save_pkl_file(file_path: Path, obj):
    try_to_find_folder_path_otherwise_make_one(file_path.parent)
    file_path = str(file_path)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except FileNotFoundError:
        file_path = re.sub('/', '//', file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)


def load_pkl_file(file_path: Path):
    file_path = str(file_path)
    if try_to_find_file(file_path) is False:
        return None
    else:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def load_exist_pkl_file_otherwise_run_and_save(file_path: Path):
    if not re.match(r".*\.pkl$", str(file_path)):
        raise Exception("'file_path' should end with '.pkl'")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(file_path, Path):
                try_to_find_folder_path_otherwise_make_one(file_path.parent)
            if load_pkl_file(file_path) is not None:
                return load_pkl_file(file_path)
            else:
                obj = func(*args, **kwargs)
                save_pkl_file(file_path, obj)
                return obj

        return wrapper

    return decorator
