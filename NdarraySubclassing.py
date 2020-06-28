import numpy as np
from typing import Iterable


class ComplexNdarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.dtype != 'complex':
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class OneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.ndim != 1:
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class IntFloatConstructedOneDimensionNdarray(np.ndarray):

    def __new__(cls, int_or_float_or_iterable):
        if isinstance(int_or_float_or_iterable, int) or isinstance(int_or_float_or_iterable, float):
            obj = np.asarray([int_or_float_or_iterable]).view(cls)
        else:
            obj = np.asarray(int_or_float_or_iterable).flatten().view(cls)
        obj = obj.astype(float)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass
