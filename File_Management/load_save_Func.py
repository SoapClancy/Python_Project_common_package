import numpy as np
import pickle
from File_Management.path_and_file_management_Func import try_to_find_file
import re


def load_npy_file(file_):
    """
    载入np文件。有的话就返回，如果没有的话则返回None
    """
    if try_to_find_file(file_) is False:
        return None
    else:
        return np.load(file_)


def save_npy_file(file_, array):
    """
    储存np文件。有的话就返回，如果没有的话则返回None
    """
    np.save(file_, array)


def load_exist_npy_file_otherwise_run_and_save(file_):
    def wrapper(func):
        if load_npy_file(file_) is not None:
            return load_npy_file(file_)
        else:
            array = func()
            save_npy_file(file_, array)
            return array

    return wrapper


def save_pkl_file(file_, obj):
    try:
        with open(file_, 'wb') as f:
            pickle.dump(obj, f)
    except FileNotFoundError:
        file_ = re.sub('/', '//', file_)
        with open(file_, 'wb') as f:
            pickle.dump(obj, f)


def load_pkl_file(file_):
    if try_to_find_file(file_) is False:
        return None
    else:
        with open(file_, "rb") as f:
            return pickle.load(f)


def load_exist_pkl_file_otherwise_run_and_save(file_):
    def wrapper(func):
        if load_pkl_file(file_) is not None:
            return load_pkl_file(file_)
        else:
            obj = func()
            save_pkl_file(file_, obj)
            return obj

    return wrapper
