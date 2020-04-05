from .format_convert_Func import datetime64_ndarray_to_datetime_tuple
from numpy import ndarray
import numpy as np
from typing import Iterable


def get_holiday_from_datetime64_ndarray(_datetime: Iterable[np.datetime64]):
    """
    返回0表示不是holiday，返回1表示是holiday
    """
    _datetime = datetime64_ndarray_to_datetime_tuple(_datetime)


def datetime_one_hot_encoder(_datetime: Iterable[np.datetime64], *,
                             including_year=False,
                             including_month=True,
                             including_day=True,
                             including_weekday=True,
                             including_hour=True,
                             including_minute=True,
                             including_second=False,
                             including_holiday=False,
                             **kwargs) -> ndarray:
    holiday = get_holiday_from_datetime64_ndarray(_datetime) if including_holiday else None
    _datetime = datetime64_ndarray_to_datetime_tuple(_datetime)
    # 从datetime.datetime中提取各种属性，编码方式：年，月，日，星期，小时，分钟，秒，是否是节假日
    datetime_iterator = np.full((_datetime.__len__(), 8), np.nan)
    for i, this_datetime_ in enumerate(_datetime):
        datetime_iterator[i, 0] = this_datetime_.year if including_year else -1
        datetime_iterator[i, 1] = this_datetime_.month if including_month else -1
        datetime_iterator[i, 2] = this_datetime_.day if including_day else -1
        datetime_iterator[i, 3] = this_datetime_.weekday() if including_weekday else -1
        datetime_iterator[i, 4] = this_datetime_.hour if including_hour else -1
        datetime_iterator[i, 5] = this_datetime_.minute if including_minute else -1
        datetime_iterator[i, 6] = this_datetime_.second if including_second else -1
        datetime_iterator[i, 7] = holiday[i] if including_holiday else -1  # 是否是节假日，1表示是，0表示不是
    # 删除无效列（i.e., 某个时间feature）
    del_col = datetime_iterator[-1, :] == -1  # 看任何一行都可以，不一定是-1行
    datetime_iterator = datetime_iterator[:, ~del_col]
    datetime_iterator = datetime_iterator.astype('int')
    # 每个col提取unique值并排序
    col_sorted_unique = []
    for col in range(datetime_iterator.shape[1]):
        col_sorted_unique.append(sorted(np.unique(datetime_iterator[:, col]), reverse=True))
    # one hot 编码
    one_hot_dims = [len(this_col_sorted_unique) for this_col_sorted_unique in col_sorted_unique]
    one_hot_results = np.full((_datetime.__len__(), sum(one_hot_dims)), 0)
    for i, this_datetime_iterator in enumerate(datetime_iterator):
        for j in range(one_hot_dims.__len__()):
            encoding_idx = np.where(this_datetime_iterator[j] == col_sorted_unique[j])[0]
            if j == 0:
                one_hot_results[i, encoding_idx] = 1
            else:
                one_hot_results[i, sum(one_hot_dims[:j]) + encoding_idx] = 1
    return one_hot_results
