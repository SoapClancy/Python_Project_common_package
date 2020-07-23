import time

from .format_convert_Func import datetime64_ndarray_to_datetime_tuple
from numpy import ndarray
import numpy as np
from typing import Iterable, Union
import pandas as pd
from pandas import DataFrame
from itertools import product
from datetime import datetime
import copy
from datetime import date


def get_holiday_from_datetime64_ndarray(_datetime: Iterable[np.datetime64]):
    """
    返回0表示不是holiday，返回1表示是holiday
    """
    raise Exception('Not implemented yet')
    # _datetime = datetime64_ndarray_to_datetime_tuple(_datetime)
    # return


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


class DatetimeOnehotEncoder:
    __slots__ = ('encoding_df_template',)

    def __init__(self, to_encoding_args=('month', 'day', 'weekday', 'holiday', 'hour', 'minute', 'summer_time')):
        """
        设置哪些变量需要被encode，可选包括：
        'month' 👉 12 bit，
        'day' 👉 31 bit，
        'weekday' 👉 7 bit，
        'holiday' 👉 1 bit，
        'hour' 👉 24 bit，
        'minute' 👉 60 bit，
        'second' 👉 60 bit，
        'summer_time 👉 1 bit.
        TODO：支持year。方法是让用户给定最小年和最大年，然后动态生成year对应的bit数
        e.g., to_encoding_args=('month', 'day', 'weekday', 'holiday', 'hour', 'minute', 'second')
        """
        self.encoding_df_template = self._initialise_encoding_df(to_encoding_args)

    @staticmethod
    def _initialise_encoding_df(to_encoding_args) -> DataFrame:
        # 动态初始化encoding_df
        columns = []
        for this_to_encoding_args in to_encoding_args:
            if this_to_encoding_args == 'month':
                columns.extend(list(product(('month',), range(1, 13))))  # 从1开始
            if this_to_encoding_args == 'day':
                columns.extend(list(product(('day',), range(1, 32))))  # 从1开始
            if this_to_encoding_args == 'weekday':
                columns.extend(list(product(('weekday',), range(1, 8))))  # 从1开始，实际是isoweekday，1代表Monday
            if this_to_encoding_args == 'holiday':
                columns.extend(list(product(('holiday',), [1])))
            if this_to_encoding_args == 'hour':
                columns.extend(list(product(('hour',), range(24))))
            if this_to_encoding_args == 'minute':
                columns.extend(list(product(('minute',), range(60))))
            if this_to_encoding_args == 'second':
                columns.extend(list(product(('second',), range(60))))
            if this_to_encoding_args == 'summer_time':
                columns.extend(list(product(('summer_time',), [1])))
        encoding_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        return encoding_df

    def __call__(self, datetime_like: pd.DatetimeIndex,
                 tz=None,
                 country=None) -> DataFrame:
        """
        输入typing中指定格式的包含datetime信息的对象，返回包含one hot encoder结果的DataFrame
        所有的输入会被转成Tuple[datetime,...]然后loop
        """
        # 初始化numpy数据
        encoding_df = np.full((datetime_like.shape[0], self.encoding_df_template.shape[1]), 0, dtype=int)
        # 把索引算出来
        required_dim_index = dict()
        for this_datetime_dim in self.encoding_df_template.columns.levels[0]:
            if this_datetime_dim != 'weekday':
                if (this_datetime_dim != 'holiday') and (this_datetime_dim != 'summer_time'):
                    required_dim_index.setdefault(this_datetime_dim, datetime_like.__getattribute__(this_datetime_dim))
                elif this_datetime_dim == 'summer_time':
                    summer_time_results = np.array(list(map(lambda x: 1 if x.dst() else 0,
                                                            datetime_like)))
                    required_dim_index.setdefault(this_datetime_dim, summer_time_results)
                else:
                    holiday_results = np.array(list(map(lambda x: country.is_holiday(x), datetime_like)))
                    required_dim_index.setdefault(this_datetime_dim, holiday_results)
            else:
                required_dim_index.setdefault(this_datetime_dim, datetime_like.__getattribute__(this_datetime_dim) + 1)
        # 写入encoding_df
        for i, this_dim_name in enumerate(self.encoding_df_template.columns):
            # 取得这一列对应的boolean数组并转成int
            this_dim = np.array(required_dim_index[this_dim_name[0]] == this_dim_name[1], dtype=int)  # type: ndarray
            encoding_df[:, i] = this_dim
        # 写入pd.DataFrame
        encoding_df = pd.DataFrame(encoding_df, columns=self.encoding_df_template.columns)
        return encoding_df


def find_nearest_datetime_idx_in_datetime_iterable(datetime_iterable: Iterable[datetime],
                                                   datetime_to_find: datetime) -> int:
    datetime_to_find = time.mktime(datetime_to_find.timetuple())
    date_time_delta = np.array([(time.mktime(x.timetuple()) - datetime_to_find) for x in datetime_iterable])
    return int(np.argmin(np.abs(date_time_delta)))