import datetime
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple, \
    find_nearest_datetime_idx_in_datetime_iterable
from Ploting import fast_plot_Func
from typing import Tuple, Union, Callable
import numpy as np
import pandas as pd
from numpy import ndarray
from Filtering.simple_filtering_Func import change_point_outlier_by_sliding_window_and_interquartile_range, \
    linear_series_outlier
from enum import Enum


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
        fast_plot_Func.time_series(x=self.synchronous_data['time'].iloc[
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
