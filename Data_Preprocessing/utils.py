import numpy as np
from numpy import ndarray
from ConvenientDataType import OneDimensionNdarray


def scale_to_ymin_ymax(x: ndarray, ymin, ymax):
    x = OneDimensionNdarray(x)
    _max, _min = np.max(x), np.min(x)
    y = (ymax - ymin) * (x - _min) / (_max - _min) + ymin
    return y


def scale_to_minus_plus_one(x: ndarray):
    return scale_to_ymin_ymax(x, -1, 1)
