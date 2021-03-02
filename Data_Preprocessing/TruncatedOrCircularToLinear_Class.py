from numpy import ndarray
import numpy as np
from typing import Union
import warnings
from Data_Preprocessing.float_precision_control_Func import float_eps
from ConvenientDataType import OneDimensionNdarray
import math


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
    __slots__ = ('period',)

    def __init__(self, period: Union[float, int]):
        self.period = period

    def transform(self, x: Union[ndarray, int, float]):
        # warnings.warn("Note that x should be in original unit", UserWarning)
        return {'cos': np.cos(math.tau * x / self.period),
                'sin': np.sin(math.tau * x / self.period)}

    def inverse_transform(self, sin_val: Union[ndarray, int, float],
                          cos_val: Union[ndarray, int, float]) -> Union[ndarray, int, float]:
        ans = np.array(np.arctan2(sin_val, cos_val))
        mask = ans < 0
        ans[mask] = math.tau + ans[mask]
        return ans * self.period / math.tau

    def __call__(self, x: ndarray):
        return self.transform(x)


# if __name__ == "__main__":
#     # Test
#     test_CircularToLinear_obj = CircularToLinear(period=360)
#     for test_angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
#         _cos = test_CircularToLinear_obj.transform(test_angle)["cos"]
#         _sin = test_CircularToLinear_obj.transform(test_angle)["sin"]
#         print(f"test_angle = {test_angle}")
#         print(f"_cos = {_cos}, _sin = {_sin}")
#         _inverse = test_CircularToLinear_obj.inverse_transform(_sin, _cos)
#         print(f"_inverse = {_inverse}")
#         print("\n")
#         assert np.isclose(test_angle, _inverse)
