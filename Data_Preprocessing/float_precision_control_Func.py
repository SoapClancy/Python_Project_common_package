import numpy as np
from numpy import ndarray
import decimal
from decimal import Decimal
from Data_Preprocessing import float_eps
import copy
from ConvenientDataType import OneDimensionNdarray, StrOneDimensionNdarray
from typing import Tuple


def covert_to_str_one_dimensional_ndarray(array: ndarray,
                                          exp: str) -> StrOneDimensionNdarray:
    """
    Use Decimal to control the floating issues. Basic idea can be explained based on following example:
    In: str(decimal.Decimal(1/3).quantize(decimal.Decimal('0.01')))
    Out: '0.33'
    """
    array = OneDimensionNdarray(array)

    def transform(x):
        results = []
        for i in x:
            results.append(str(Decimal(i).quantize(Decimal(exp))))
        return np.array(results)

    transform_vectorised = np.vectorize(transform, signature='(n)->(n)')
    return transform_vectorised(array)


def convert_float_to_arbitrary_precision(num: float, decimal_places: int) -> float:
    """
    将一个float转成任意精度
    :param num: float number
    :param decimal_places: 表示的是精度。e.g., 2表示精确到小数点后2位
    :return:
    """
    return round(num, decimal_places)


def get_decimal_places_of_float(num: float) -> int:
    """
    得到一个float数的小数点后面有几位
    :param num: float number
    :return: 位数
    """
    error = np.inf
    dp = 0
    while error > float_eps * 10:
        error = abs(num - convert_float_to_arbitrary_precision(num, dp))
        dp += 1
    return dp - 1


def convert_ndarray_to_arbitrary_precision(array: ndarray, decimal_places: int) -> ndarray:
    return np.around(array, decimal_places)


def limit_ndarray_max_and_min(array: ndarray, min_value: float, max_value: float):
    new_array = copy.deepcopy(array)
    new_array[new_array < min_value] = min_value
    new_array[new_array > max_value] = max_value
    return new_array
