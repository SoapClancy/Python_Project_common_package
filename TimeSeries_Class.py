import datetime
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple
from Time_Processing.datetime_utils import find_nearest_datetime_idx_in_datetime_iterable
from Ploting.fast_plot_Func import time_series, series
from typing import Tuple, Union, Callable
import numpy as np
import pandas as pd
from numpy import ndarray
from Filtering.simple_filtering_Func import change_point_outlier_by_sliding_window_and_interquartile_range, \
    linear_series_outlier
from enum import Enum
import copy
import isoweek
from Time_Processing.datetime_utils import find_nearest_datetime_idx_in_datetime_iterable
from scipy.signal import get_window


def merge_two_time_series_df(main_time_series_df: pd.DataFrame,
                             new_time_series_df: pd.DataFrame,
                             naively_ignore_tz_info: bool = True,
                             resolution: str = 'second',
                             do_interpolate: bool = True,
                             interpolate_method='time') -> pd.DataFrame:
    """
    用于合并两个two_time_series_df。two_time_series_df指的是以datetime作为index的pd.DataFrame。
    方法在于分别从两个df的index中提取出year, month, day, hour, minute, second作为新的columns，再inner merge
    :param main_time_series_df: the result's index should be the same as the index of main_time_series_df
    :param new_time_series_df: the time_series_df can be considered as extra information/dimensions added
    :param naively_ignore_tz_info: 不用考虑tz转换！df自带的merge如果遇到一个df的index有tz而另一个df的index没有tz就没办法合并，
    而且转成有tz的index有可能因为DST造成ambiguous exception
    :param resolution: 合并的resolution，目前只支持到second
    :param do_interpolate
    :param interpolate_method
    :return:
    """
    main_time_series_df = copy.deepcopy(main_time_series_df)
    new_time_series_df = copy.deepcopy(new_time_series_df)

    def merge_existing_df_datetime_df(existing_df: pd.DataFrame):
        index = existing_df.index
        datetime_df = pd.DataFrame(index=index,
                                   data={'year': index.year.values,
                                         'month': index.month.values,
                                         'day': index.day.values,
                                         'hour': index.hour.values,
                                         'minute': index.minute.values,
                                         'second': index.second.values})
        return existing_df.join(datetime_df)

    # 预处理，生成新df
    merge_main_time_series_df_datetime_df = merge_existing_df_datetime_df(main_time_series_df)
    merge_new_time_series_df_datetime_df = merge_existing_df_datetime_df(new_time_series_df)

    # new_time_series_df精度比main_time_series_df低的情况
    if (new_time_series_df.index[1] - new_time_series_df.index[0]) > \
            (main_time_series_df.index[1] - main_time_series_df.index[0]):
        merge_main_time_series_df_new_time_series_df = pd.merge(merge_main_time_series_df_datetime_df,
                                                                merge_new_time_series_df_datetime_df,
                                                                on=['year', 'month', 'day', 'hour', 'minute', 'second'],
                                                                how='left')
        merge_main_time_series_df_new_time_series_df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'],
                                                          inplace=True)
        merge_main_time_series_df_new_time_series_df.set_index(main_time_series_df.index, inplace=True)
        if do_interpolate:
            merge_main_time_series_df_new_time_series_df.interpolate(method=interpolate_method, inplace=True)
    # TODO new_time_series_df精度比main_time_series_df高的情况。这里需要aggregate
    else:
        raise
    return merge_main_time_series_df_new_time_series_df


class TimeSeries(pd.DataFrame):
    __slots__ = ()

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        return pd.DataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series

    def _check_ordinal_time_delta(self):
        # 检查每两个记录之间的time delta是否相等，否则raise
        delta = self.index[1:] - self.index[:-1]
        if (delta.max() - delta.min()) > datetime.timedelta(seconds=1):
            raise Exception('data的index的间隔不是一个常数')

    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.index, pd.DatetimeIndex):
            raise Exception("Time series data must use pd.DatetimeIndex as index")
        self._check_ordinal_time_delta()

    def view_as_dataframe(self):
        """
        配合pycharm的scientific mode显示
        """
        return pd.DataFrame(self)

    def __repr__(self):
        return f'Time series from {self.index[0]} to {self.index[-1]}, ' \
               f'resolution = {self.adjacent_recordings_timedelta}'

    @property
    def adjacent_recordings_timedelta(self) -> datetime.timedelta:
        return self.index[1] - self.index[0]

    @property
    def number_of_recordings_per_day(self):
        number = int(24 * 3600 / self.adjacent_recordings_timedelta.seconds)
        if number == 0:
            raise Exception('每天的记录数为0')
        return number

    @property
    def number_of_recordings_per_week(self):
        number = int(24 * 3600 * 7 / self.adjacent_recordings_timedelta.seconds)
        if number == 0:
            raise Exception('每周的记录数为0')
        return number

    def to_windowed_time_series(self, *, window_length: datetime.timedelta, window: str = None):
        return WindowedTimeSeries(self, window_length=window_length, window=window)


class WindowedTimeSeries(TimeSeries):
    __slots__ = ('window_interval', 'window', 'window_length', '__iter_count')

    def __init__(self, *args, window_length: datetime.timedelta, window: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.window_length = window_length
        setattr(self, 'window_interval', self._cal_window_interval())

    def __repr__(self):
        return f'Windowed time series from {self.index[0]} to {self.index[-1]}, ' \
               f'resolution = {self.adjacent_recordings_timedelta}, ' \
               f'window = {self.window}, ' \
               f'window_length = {self.window_length}'

    def _cal_window_interval(self) -> tuple:
        """
        计算window两个边界对应的索引
        """
        self_index_as_asi8 = self.index.asi8
        window_interval = [self.index[0]]
        while True:
            window_interval_next = window_interval[-1] + self.window_length
            if window_interval_next > self.index[-1]:
                window_interval.append(self.index[-1])
                break
            else:
                # 找到self.index中距离window_interval_next中最近的一个，防止__getitem__找不到对应的index
                if window_interval_next not in self.index:
                    window_interval_next = self.index[np.argmin(np.abs(self_index_as_asi8 -
                                                                       window_interval_next.value))]
                window_interval.append(window_interval_next)
        return tuple(window_interval)

    def __getitem__(self, index: int) -> TimeSeries:
        windowed_data = self.loc[self.window_interval[index]:self.window_interval[index + 1]].iloc[:-1]
        # 增加必要的窗函数
        if self.window is not None:
            scipy_window = get_window(self.window, windowed_data.__len__()).reshape(-1, 1)
            windowed_data *= np.tile(scipy_window, (1, windowed_data.shape[-1]))
        return windowed_data

    def __iter__(self):
        self.__iter_count = 0  # reset，以便以后继续能iter
        return self

    def __next__(self) -> TimeSeries:
        try:
            this_window_data = self[self.__iter_count]
            self.__iter_count += 1
        except IndexError:
            raise StopIteration
        return this_window_data


class UnivariateTimeSeries(TimeSeries):
    def detrend(self,
                resample_args_dict: dict, *,
                inplace: bool = False) -> pd.DataFrame:  # e.g., resample_args_dict={'rule': '24H'}
        # downsample
        trend = self.resample(**resample_args_dict).mean()
        detrended_data = merge_two_time_series_df(self, trend, do_interpolate=False)
        detrended_data = detrended_data.fillna(method='pad', axis=0)
        detrended_data.iloc[:, 0] = detrended_data.iloc[:, 0] - detrended_data.iloc[:, 1]  # 只有两列
        detrended_data = detrended_data.iloc[:, [0]]
        return detrended_data

    def plot_group_by_week(self, **kwargs):
        # 补齐第一周
        missing_recording_number = self.index.weekday[0] * self.number_of_recordings_per_day
        first_missing_week = pd.DataFrame(np.nan,
                                          columns=self.columns,
                                          index=pd.date_range(
                                              end=self.index[0] - self.adjacent_recordings_timedelta,
                                              periods=missing_recording_number,
                                              freq=self.adjacent_recordings_timedelta))
        # 补齐最后一周
        missing_recording_number = (6 - self.index.weekday[-1]) * self.number_of_recordings_per_day
        last_missing_week = pd.DataFrame(np.nan,
                                         columns=self.columns,
                                         index=pd.date_range(
                                             start=self.index[-1] + self.adjacent_recordings_timedelta,
                                             periods=missing_recording_number,
                                             freq=self.adjacent_recordings_timedelta))

        # 补齐
        data_extend = pd.concat((first_missing_week,
                                 self,
                                 last_missing_week))
        total_number_of_weeks = int(data_extend.__len__() / self.number_of_recordings_per_week)
        data_extend['week_no_in_the_dataset'] = np.arange(total_number_of_weeks).repeat(
            self.number_of_recordings_per_week)
        # group by
        data_extend = data_extend.groupby('week_no_in_the_dataset')
        # %% 画图
        ax = None
        average = []
        for _, this_week_data in data_extend:
            this_week_data = this_week_data.reindex(
                columns=this_week_data.columns.difference(['week_no_in_the_dataset'])).values.flatten()
            average.append(this_week_data)
            ax = series(range(0, self.number_of_recordings_per_week),
                        this_week_data, ax=ax, color='b',
                        figure_size=(10, 2.4))
        average = np.array(average)
        average[np.isinf(average)] = np.nan
        ax = series(range(0, self.number_of_recordings_per_week),
                    np.nanmean(average, axis=0),
                    ax=ax, color='r',
                    label='Mean value',
                    figure_size=(10, 2.4),
                    x_lim=(-1, self.number_of_recordings_per_week),
                    x_ticks=(tuple(range(0, self.number_of_recordings_per_week, self.number_of_recordings_per_day)),
                             ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                    **kwargs)
        return ax


class SynchronousTimeSeriesData:
    __slots__ = ('synchronous_data', 'data_category', 'data_category_detailed')

    def __init__(self, synchronous_data: pd.DataFrame, data_category: ndarray = None,
                 category_detailed: pd.DataFrame = None):
        if ('time' not in synchronous_data.columns) and ('Time' not in synchronous_data.columns):
            raise Exception("'synchronous_data' should be a pd.DataFrame with 'time' or 'Time' column")
        if 'Time' in synchronous_data.columns:
            synchronous_data = synchronous_data.rename(columns={'Time': 'time'})
        self.synchronous_data = synchronous_data  # type: pd.DataFrame
        self.data_category = data_category  # type: ndarray
        self.data_category_detailed = category_detailed  # type: pd.DataFrame

    def down_sample(self, aggregate_on_sample_number: int, aggregate_on_category: Tuple[int, ...] = (0,),
                    category_is_outlier: bool = True):
        """
        将数据精度降低。
        每一次aggregation，只考虑time window里面有所有符合aggregate_on_category标准的有效记录，求和再除以有效记录的个数。
        主要的麻烦点在于处理好data_category和data_category_detailed。
        · 一般来说已经没有意义，需要重新分类，所以将data_category和data_category_detailed置为none
        · 如果category_is_outlier，那么data_category和data_category_detailed的意义在于：0代表该数据有效，-1代表nan
        :param aggregate_on_sample_number:
        :param aggregate_on_category:
        :param category_is_outlier:
        :return:
        """
        synchronous_data = {key: [] for key in self.synchronous_data.columns}
        data_category_detailed = {key: [] for key in self.synchronous_data.columns}

        for i, this_key in enumerate(self.synchronous_data.columns):
            if this_key == 'time':
                for time_window_start_idx in range(0, self.data_category.__len__(), aggregate_on_sample_number):
                    # time维度简单，只要取中间值就好
                    if aggregate_on_sample_number % 2 == 1:
                        time_window_mean_time = self.synchronous_data.iloc[
                            time_window_start_idx + int(aggregate_on_sample_number / 2)][
                            'time']
                    else:
                        time_window_mean_time_1 = self.synchronous_data.iloc[
                            time_window_start_idx + int(aggregate_on_sample_number / 2) - 1][
                            'time']
                        time_window_mean_time_2 = self.synchronous_data.iloc[
                            time_window_start_idx + int(aggregate_on_sample_number / 2)][
                            'time']
                        time_window_mean_time = time_window_mean_time_1 + (
                                time_window_mean_time_2 - time_window_mean_time_1) / 2

                    synchronous_data['time'].append(time_window_mean_time)
                    data_category_detailed['time'].append(0)
            else:
                this_dim_data = self.synchronous_data[this_key].values
                this_dim_category_detailed = self.data_category_detailed[this_key].values
                # reshape
                this_dim_data = this_dim_data.reshape(-1, aggregate_on_sample_number)
                this_dim_category_detailed = this_dim_category_detailed.reshape(-1, aggregate_on_sample_number)
                # 将this_dim_category_detailed中符合aggregate_on_category标准的置为1， 否则为0
                aggregate_on_flag = np.full(this_dim_category_detailed.shape, True)
                for this_aggregate_on_category in aggregate_on_category:
                    aggregate_on_flag = np.bitwise_and(aggregate_on_flag,
                                                       this_dim_category_detailed == this_aggregate_on_category)
                this_dim_category_detailed[aggregate_on_flag] = 1
                this_dim_category_detailed[~aggregate_on_flag] = 0
                # 选用符合this_aggregate_on_category的数据进行aggregate
                good_data_number_in_this_window = np.sum(this_dim_category_detailed, axis=1)
                synchronous_data[this_key] = np.nansum(
                    this_dim_data * this_dim_category_detailed, axis=1) / good_data_number_in_this_window

                # 考虑data_category_detailed变量
                if not category_is_outlier:  # 极少可能会这样...这种情况下前面的程序已经多算了
                    data_category_detailed[this_key] = None
                else:
                    data_category_detailed[this_key] = good_data_number_in_this_window
                    all_good_flag = good_data_number_in_this_window == aggregate_on_sample_number
                    data_category_detailed[this_key][all_good_flag] = 0
                    data_category_detailed[this_key][~all_good_flag] = 1

        # 考虑data_category变量
        if not category_is_outlier:  # 极少可能会这样...这种情况下前面的程序已经多算了
            data_category = None
        else:
            self.data_category = self.data_category.reshape(-1, aggregate_on_sample_number)
            data_category = np.full((self.data_category.shape[0],), np.nan)

            good_data_flag = np.full(self.data_category.shape, True)
            for this_aggregate_on_category in aggregate_on_category:
                good_data_flag = np.bitwise_and(good_data_flag,
                                                self.data_category == this_aggregate_on_category)
            good_data_number = np.sum(good_data_flag, axis=1)
            data_category[good_data_number == aggregate_on_sample_number] = 0
            data_category[good_data_number != aggregate_on_sample_number] = -1

        self.synchronous_data = pd.DataFrame(synchronous_data)
        self.data_category = data_category
        self.data_category_detailed = pd.DataFrame(data_category_detailed)

        return self.synchronous_data, self.data_category, self.data_category_detailed

    def find_truncate_idx(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        datetime_all = datetime64_ndarray_to_datetime_tuple(self.synchronous_data['time'].values)
        start_time = start_time or datetime_all[0]  # type: datetime.datetime
        end_time = end_time or datetime_all[-1]  # type: datetime.datetime
        start_time_idx = find_nearest_datetime_idx_in_datetime_iterable(datetime_all, start_time)
        end_time_idx = find_nearest_datetime_idx_in_datetime_iterable(datetime_all, end_time)
        return start_time_idx, end_time_idx

    def do_truncate(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        start_time_idx, end_time_idx = self.find_truncate_idx(start_time=start_time, end_time=end_time)
        self.synchronous_data = self.synchronous_data.iloc[start_time_idx:end_time_idx + 1]
        self.data_category_detailed = self.data_category_detailed.iloc[start_time_idx:end_time_idx + 1]
        self.data_category = self.data_category[start_time_idx:end_time_idx + 1]
        return self.synchronous_data, self.data_category, self.data_category_detailed

    def get_current_season(self, season_template) -> Union[tuple, str]:
        month = np.array([x.month for x in datetime64_ndarray_to_datetime_tuple(
            self.synchronous_data['time'].values)])
        current_season = []
        for member in season_template.__members__:
            for i in season_template[member].value:
                if i in month:
                    current_season.append(season_template[member].name)
        if all(('spring' in current_season, 'summer' in current_season,
                'autumn' in current_season, 'winter' in current_season)):
            return 'all seasons'
        else:
            return tuple(set(current_season))

    def find_truncate_by_season_mask(self, season_to_be_queried: str, season_template: Enum):
        month = np.array([x.month for x in datetime64_ndarray_to_datetime_tuple(
            self.synchronous_data['time'].values)])
        truncate_by_season_mask = np.full(month.size, False)
        for i in season_template[season_to_be_queried].value:
            truncate_by_season_mask = np.bitwise_or(truncate_by_season_mask,
                                                    month == i)
        return truncate_by_season_mask

    def do_truncate_by_season(self, season_to_be_queried: str, season_template: Enum):
        truncate_by_season_mask = self.find_truncate_by_season_mask(season_to_be_queried, season_template)
        self.synchronous_data = self.synchronous_data[truncate_by_season_mask]
        self.data_category_detailed = self.data_category_detailed[truncate_by_season_mask]
        self.data_category = self.data_category[truncate_by_season_mask]
        return self.synchronous_data, self.data_category, self.data_category_detailed

    def plot_time_stamp_to_tuple_synchronous_data(self, synchronous_data_name: Tuple[str, ...], *,
                                                  start_time: datetime.datetime = None,
                                                  end_time: datetime.datetime = None,
                                                  **kwargs):
        """
        画出在指定时间内的指定time series
        """
        start_time_idx, end_time_idx = self.find_truncate_idx(start_time=start_time, end_time=end_time)
        time_series(x=self.synchronous_data['time'].iloc[
                      start_time_idx:end_time_idx].values,
                    y=self.synchronous_data[[*synchronous_data_name]].iloc[
                      start_time_idx:end_time_idx].values, **kwargs)

    def __identify_outliers_in_tuple_synchronous_data(self, func: Callable, synchronous_data_name: Tuple[str, ...],
                                                      **kwargs):
        outlier_mask = np.full(self.synchronous_data.shape[0], False)

        def single_identify(this_synchronous_data_name: str):
            nonlocal outlier_mask
            temp = func(self.synchronous_data[this_synchronous_data_name].values, **kwargs)
            outlier_mask = np.bitwise_or(outlier_mask, temp)

        for this_name in synchronous_data_name:
            single_identify(this_name)
        return outlier_mask

    def missing_data_outlier_in_tuple_synchronous_data(self, synchronous_data_name: Tuple[str, ...]) -> ndarray:
        """
        missing data are regarded as outlier
        """
        return self.__identify_outliers_in_tuple_synchronous_data(np.isnan, synchronous_data_name)

    def change_point_outliers_in_tuple_synchronous_data(self, synchronous_data_name: Tuple[str, ...],
                                                        sliding_window_back_or_forward_sample):
        """
        change point outlier in series data
        """
        return self.__identify_outliers_in_tuple_synchronous_data(
            change_point_outlier_by_sliding_window_and_interquartile_range, synchronous_data_name,
            sliding_window_back_or_forward_sample=sliding_window_back_or_forward_sample)

    def linear_series_outliers_in_tuple_synchronous_data(self, synchronous_data_name: Tuple[str, ...],
                                                         sliding_window_forward_sample: int) -> ndarray:
        """
        linear series outlier in series data
        """
        return self.__identify_outliers_in_tuple_synchronous_data(
            linear_series_outlier, synchronous_data_name,
            sliding_window_forward_sample=sliding_window_forward_sample)
