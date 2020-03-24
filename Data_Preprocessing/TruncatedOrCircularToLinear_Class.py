from numpy import ndarray
import numpy as np
from typing import Union
import warnings
from Data_Preprocessing.float_precision_control_Func import float_eps


class TruncatedToLinear:
    __slots__ = ('lower_boundary', 'upper_boundary')

    def __init__(self, lower_boundary: Union[float, int], upper_boundary: Union[float, int]):
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

    def transform(self, x: ndarray):
        if np.nanmax(x) >= self.upper_boundary:
            x[x >= self.upper_boundary] = self.upper_boundary - 10e8 * float_eps  # 强行不超过上界
            warnings.warn('At least one element >= upper_boundary. Modified according to upper_boundary')
        if np.nanmin(x) <= self.lower_boundary:
            x[x <= self.lower_boundary] = self.lower_boundary + 10e8 * float_eps
            warnings.warn('At least one element <= lower_boundary. Modified according to lower_boundary')
        return np.log((x - self.lower_boundary) / (self.upper_boundary - x))

    def inverse_transform(self, y: ndarray):
        return (np.exp(y) * self.upper_boundary + self.lower_boundary) / (1 + np.exp(y))


class CircularToLinear:
    __slots__ = ('lower_boundary', 'upper_boundary', 'period')

    def __init__(self, lower_boundary: Union[float, int], upper_boundary: Union[float, int], period: Union[float, int]):
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.period = period

    def transform(self, x: ndarray):
        return np.cos(2 * np.pi * x / self.period), np.sin(2 * np.pi * x / self.period)
