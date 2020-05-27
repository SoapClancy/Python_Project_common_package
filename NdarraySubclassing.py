import numpy as np


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

