import numpy as np
from typing import Iterable
from numpy import ndarray
import pandas as pd


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


class IntOneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        try:
            obj = OneDimensionNdarray(input_array)
            if obj.dtype != int:
                raise TypeError
        except TypeError:
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class StrOneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        try:
            obj = OneDimensionNdarray(input_array)
            if not isinstance(obj[0], str):
                raise TypeError
        except TypeError:
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


class UncertaintyDataFrame(pd.DataFrame):
    __slots__ = ()

    @property
    def _constructor(self):
        return UncertaintyDataFrame

    @property
    def _constructor_expanddim(self):
        return pd.DataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        StrOneDimensionNdarray(self.index.values)
        if self.index[-1] != 'mean':
            raise Exception("UncertaintyDataFrame must use StrOneDimensionNdarray as index, "
                            "and the last index should be 'mean'")

    def infer_higher_half_percentiles(self, lower_half_percentiles: StrOneDimensionNdarray) -> StrOneDimensionNdarray:
        """
        Use lower half percentiles to infer higher half percentiles
        """
        higher_half_percentiles = []
        for this_lower_half_percentile in lower_half_percentiles:
            this_higher_half_percentile_index = np.argmin(
                np.abs(self.index.values[:-1].astype(np.float) - (100 - float(this_lower_half_percentile)))
            )
            higher_half_percentiles.append(self.index.values[this_higher_half_percentile_index])
        return StrOneDimensionNdarray(higher_half_percentiles)
