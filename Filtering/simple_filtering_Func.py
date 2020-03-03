import numpy as np
from numpy import ndarray
from typing import Tuple, Union
from Data_Preprocessing import float_eps
import copy


def interquartile_range(x: ndarray, axis=1):
    """
    标准的interquartile检测outlier的方法
    :param x: 待检测的数据
    :param axis: 待检测的数据
    :return: outlier的值的上下界
    """
    if x.ndim == 1:
        q25, q75 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    else:
        q25, q75 = np.nanpercentile(x, 25, axis=axis), np.nanpercentile(x, 75, axis=axis)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    return lower, upper


def interquartile_outlier(x: ndarray):
    """
    只适用于1维ndarray的快速interquartile outlier判别方法
    :param x:
    :return: 布尔数组，True表示outlier
    """
    lower, upper = interquartile_range(x)
    return np.bitwise_or(x < lower, x > upper)


def shut_down_outlier(*, predictor_var: ndarray, dependent_var: ndarray,
                      cannot_be_zero_predictor_var_range: Tuple[float, float],
                      zero_upper_tolerance: float = float_eps,
                      zero_upper_tolerance_factor: float = None) -> ndarray:
    """
    用于检测shut down outliers：在一定的自变量范围内，因变量不应该是0或者非常接近0
    :param predictor_var: 自变量
    :param dependent_var: 因变量
    :param cannot_be_zero_predictor_var_range: 一个tuple，表示的是因变量不应该是0或者非常接近0的范围（左闭右开）
    :param zero_upper_tolerance: 接近0
    :param zero_upper_tolerance_factor：线性增加地接近0
    :return: 布尔数组，True表示outlier
    """
    range_idx = np.bitwise_and(predictor_var >= cannot_be_zero_predictor_var_range[0],
                               predictor_var < cannot_be_zero_predictor_var_range[1])
    if zero_upper_tolerance_factor is None:
        zero_idx = dependent_var < zero_upper_tolerance
    else:
        zero_idx = dependent_var < (predictor_var * zero_upper_tolerance_factor)
    return np.bitwise_and(range_idx, zero_idx)


def change_point_outlier_by_sliding_window_and_interquartile_range(data_series: ndarray,
                                                                   sliding_window_back_or_forward_sample: int):
    """
    分析一个series中的突变值
    :param data_series:
    :param sliding_window_back_or_forward_sample:
    :return: True表示突变值的布尔数组
    """
    # 构建sliding window数组
    data_series_copy = copy.deepcopy(data_series)
    sliding_data = np.full((sliding_window_back_or_forward_sample * 2 + 1, data_series.size), np.nan)
    for row, i in enumerate(range(-sliding_window_back_or_forward_sample, sliding_window_back_or_forward_sample + 1)):
        sliding_data[row, :] = np.roll(data_series_copy, -i)
    sliding_data[:, 0:sliding_window_back_or_forward_sample] = np.nan
    sliding_data[:, -sliding_window_back_or_forward_sample:] = np.nan
    # 通过sliding好的数据判断每个window里的outlier情况
    outlier = interquartile_range(sliding_data.T)
    return np.bitwise_or(data_series_copy < outlier[0], data_series_copy > outlier[1])


def linear_series_outlier(data_series: ndarray, sliding_window_forward_sample: int, error: float = None):
    """
    检测一个ndarray序列（一维）中的linear变化的pattern
    :return: linear变化的pattern的索引的布尔数组
    """
    series_0_delta = np.abs(data_series[:-1] - data_series[1:])
    sliding_data = np.full((sliding_window_forward_sample, data_series.size), np.nan)
    sliding_data[0, :-1] = series_0_delta
    for i in range(1, sliding_window_forward_sample):
        sliding_data[i, :-1 - i] = series_0_delta[i:]
    sliding_data_min = np.nanmin(sliding_data, axis=0)
    sliding_data_max = np.nanmax(sliding_data, axis=0)
    constant = np.abs(sliding_data_max - sliding_data_min) < (error or float_eps * 10)
    temp = copy.deepcopy(constant)
    for i in range(1, sliding_window_forward_sample + 1):
        roll = np.roll(temp, i)
        roll[:i] = False
        constant = np.bitwise_or(constant, roll)
    return constant


def out_of_range_outlier(data_series: ndarray, lower_bound: Union[float, int], upper_bound: Union[float, int]):
    return np.bitwise_or(data_series > upper_bound, data_series < lower_bound)
