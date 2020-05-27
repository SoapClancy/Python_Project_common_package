from FFT_Class import FFTProcessor, APFormFourierSeriesProcessor, FourierSeriesProcessor
from TimeSeries_Class import merge_two_time_series_df, TimeSeries, WindowedTimeSeries
import pandas as pd
from numpy import ndarray
from Ploting.fast_plot_Func import *
from typing import Tuple


class FFTCorrelationMeta(type):
    @staticmethod
    def _make_init(fields):
        code = f"def __init__(cls, *, {', '.join(fields)}, **kwargs):\n"
        for this_filed in fields:
            code += f'    cls.{this_filed} = {this_filed}\n'
        return code

    def __new__(mcs, name, bases, clsdict):
        if '__init__' not in clsdict:
            init_fields = ['correlation_func',
                           'n_fft',
                           'considered_frequency_unit']
            exec(mcs._make_init(init_fields), globals(), clsdict)
        clsobj = super().__new__(mcs, name, bases,
                                 clsdict)
        return clsobj


class FFTCorrelation(metaclass=FFTCorrelationMeta):
    __slots__ = (
        'time_series',
        'correlation_func',
        'n_fft',
        'considered_frequency_unit'
    )

    @property
    def sampling_period(self):
        return (self.time_series.index[1] - self.time_series.index[0]).seconds


class BivariateFFTCorrelation(FFTCorrelation):
    __slots__ = ('main_fft',
                 'vice_fft',
                 'main_found_peaks',
                 'vice_found_peaks',
                 'main_ifft',
                 'vice_ifft')

    def __init__(self, *,
                 _time_series: TimeSeries = None,
                 main_time_series_df: pd.DataFrame = None,
                 vice_time_series_df: pd.DataFrame = None,
                 main_considered_peaks_index: tuple,
                 vice_considered_peaks_index: tuple,
                 main_find_peaks_args: dict = None,
                 vice_find_peaks_args: dict = None,
                 **kwargs):
        """
        参数time_series和(main_time_series_df和vice_time_series_df)二选一。
        :param _time_series TimeSeries类或者其子类，默认第0列对应'main_time_series'，第1列对应'vice_time_series'

        :param main_time_series_df 主时间序列，副时间序列存在的目的是为了更好地理解它。比如，在load和temperature相关性建模中，
        load是主，temperature是副。这种关系主要影响corr_between_main_peaks_f_and_vice方法和
        corr_between_combined_main_peaks_f_and_vice方法的行为。load会作为一个完整的时序的序列。但是temperature会被fft分解

        :param vice_time_series_df 副时间序列

        :param main_considered_peaks_index main哪些分量会被考虑到（这是它们的索引）。注意，这与considered_frequency_unit
        有关，也与FFTProcessor类的find_peaks_of_fft_frequency方法有关

        :param vice_considered_peaks_index
        """
        super(BivariateFFTCorrelation, self).__init__(**kwargs)
        if (main_time_series_df is None) and (vice_time_series_df is None):
            self.time_series = _time_series  # type: TimeSeries
        else:
            self.time_series = merge_two_time_series_df(main_time_series_df,
                                                        vice_time_series_df)
        # fft
        self.main_fft, self.vice_fft = self._do_fft()
        # 找peaks
        self.main_found_peaks, self.vice_found_peaks = self._find_peaks(main_find_peaks_args=main_find_peaks_args,
                                                                        vice_find_peaks_args=vice_find_peaks_args)
        # 对考虑的分量分别单独进行ifft
        # x_ifft的key就是x_considered_peaks_index, value是一个tuple(hz频率，指定单位频率，幅值，角度，ifft的结果)
        self.main_ifft, self.vice_ifft = self._cal_ifft(main_considered_peaks_index=main_considered_peaks_index,
                                                        vice_considered_peaks_index=vice_considered_peaks_index)

    def _do_fft(self):
        # 分别进行fft
        main_time_series_fft = FFTProcessor(self.time_series.iloc[:, 0].values,
                                            sampling_period=self.sampling_period,
                                            name='main_time_series',
                                            n_fft=self.n_fft)  # type: FFTProcessor
        vice_time_series_fft = FFTProcessor(self.time_series.iloc[:, 1].values,
                                            sampling_period=self.sampling_period,
                                            name='vice_time_series',
                                            n_fft=self.n_fft)  # type: FFTProcessor
        return main_time_series_fft, vice_time_series_fft

    def _find_peaks(self, *, main_find_peaks_args: dict = None,
                    vice_find_peaks_args: dict = None) -> Tuple[tuple, tuple]:

        # 找peaks
        main_find_peaks_args = main_find_peaks_args or {}
        vice_find_peaks_args = vice_find_peaks_args or {}
        main_found_peaks = self.main_fft.find_peaks_of_fft_frequency(
            self.considered_frequency_unit,
            **main_find_peaks_args)
        vice_found_peaks = self.vice_fft.find_peaks_of_fft_frequency(
            self.considered_frequency_unit,
            **vice_find_peaks_args)
        return main_found_peaks, vice_found_peaks

    def _cal_ifft(self, main_considered_peaks_index, vice_considered_peaks_index):
        # key是int， 代表第几个peak。value是一个tuple(hz频率，指定单位频率，幅值，角度，ifft的结果)
        def one_ifft(fft_processor, considered_peaks_index, peaks_index_in_full_results: ndarray):
            """
            :param fft_processor 需要处理的FFTProcessor对象
            :param considered_peaks_index 考虑第几个peaks
            :param peaks_index_in_full_results peaks分量在full FFT results中的索引。即scipy.signal.find_peaks函数的返回值0
            """
            temp = {key: None for key in considered_peaks_index}
            for key in temp:
                fft_results = fft_processor.single_sided_frequency_axis_all_supported()[
                    [self.considered_frequency_unit, 'magnitude', 'phase angle (rad)']]
                this_peaks_index_in_full_results = peaks_index_in_full_results[key]
                hz_f = fft_results.index[this_peaks_index_in_full_results]
                magnitude = fft_results.values[this_peaks_index_in_full_results, 1]
                phase = fft_results.values[this_peaks_index_in_full_results, 2]
                re_constructed_time_domain = APFormFourierSeriesProcessor(frequency=np.array([hz_f]),
                                                                          magnitude=np.array([magnitude]),
                                                                          phase=np.array([phase]))
                re_constructed_time_domain = re_constructed_time_domain(self.time_series.index)
                temp[key] = (hz_f,
                             fft_results.values[0],
                             magnitude,
                             phase,
                             re_constructed_time_domain)
            return temp

        main_ifft = one_ifft(self.main_fft, main_considered_peaks_index, self.main_found_peaks[1])
        vice_ifft = one_ifft(self.vice_fft, vice_considered_peaks_index, self.vice_found_peaks[1])
        return main_ifft, vice_ifft

    def corr_between_pairwise_peaks_f(self) -> ndarray:
        pass

    def corr_between_main_peaks_f_and_vice(self) -> ndarray:
        pass

    def corr_between_combined_main_peaks_f_and_vice(self) -> ndarray:
        pass