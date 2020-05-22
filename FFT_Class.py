import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fft
from datetime import timedelta
from typing import Union, Tuple
from Ploting.fast_plot_Func import *
from enum import Enum, unique
from pathlib import Path
from Writting.utils import put_cached_png_into_a_docx
from scipy.signal import stft, find_peaks
from NdarraySubclassing import ComplexNdarray
from Ploting.utils import BufferedFigureSaver
from TimeSeries_Class import WindowedTimeSeries
from Ploting.adjust_Func import adjust_lim_label_ticks


class FFTProcessor:
    __slots__ = ('original_signal', 'sampling_period', 'name', 'n_fft')

    @unique
    class SupportedTransformedPeriod(Enum):
        # 命名方式是：
        # ('convenient_period_unit_name',
        # 'convenient_frequency_unit_name',
        # 'plot_x_lim', 👉 ()代表不设置限制让matplotlib自动决定，
        # (x1, x2)代表一个图范围是x1到x2，((x1,x2), (x3,x4))代表两个图
        # 'scale_factor')
        second = ('second', '1/second', (0, None), 1)
        minute = ('minute', '1/minute', (-10e-4, None), 60)
        hour = ('hour', '1/hour', ((-10e-4, 0.3), (-10e-4, None)), 60 * 60)
        half_day = ('half day', '1/half day', ((-0.05, 4), (-0.05, None)), 60 * 60 * 12)
        day = ('day', '1/day', ((-0.1, 12), (-0.1, None)), 60 * 60 * 24)
        week = ('week', '1/week', (3.5, 10.5), 60 * 60 * 24 * 7)
        _28_days = ('28 days', '1/28 days', ((-0.25, 100), (-0.25, None)), 60 * 60 * 24 * 28)
        _364_days = ('364 days', '1/364 days', ((-0.25, 30.5), (359.5, 375.5)), 60 * 60 * 24 * 364)
        _365_days = ('365 days', '1/365 days', ((-0.25, 30.5), (359.5, 375.5)), 60 * 60 * 24 * 365)
        _365_25_days = ('365.25 days', '1/365.25 days', ((-0.25, 30.5), (359.5, 375.5)), 60 * 60 * 24 * 365.25)

        @classmethod
        def get_by_convenient_frequency_unit_name(cls, convenient_frequency_unit_name: str):
            for value in cls:
                if convenient_frequency_unit_name == value.value[1]:
                    return value.value
            raise Exception('Unsupported transformed frequency')

        @classmethod
        def get_by_convenient_period_unit_name(cls, convenient_period_unit_name: str):
            for value in cls:
                if convenient_period_unit_name == value.value[0]:
                    return value.value
            raise Exception('Unsupported transformed period')

        @classmethod
        def list_all_convenient_period_unit_names(cls) -> tuple:
            return tuple([x.value[0] for x in cls])

        @classmethod
        def list_all_convenient_frequency_unit_names(cls) -> tuple:
            return tuple([x.value[1] for x in cls])

    def __init__(self, original_signal: ndarray, *,
                 sampling_period: Union[int, float],
                 name: str,
                 n_fft: int = None):
        if original_signal.ndim > 1:
            raise Exception('Only consider 1-D data')
        self.original_signal = original_signal
        self.sampling_period = sampling_period
        self.name = name
        self.n_fft = n_fft or self.length_of_signal

    def __str__(self):
        return self.name

    @property
    def sampling_frequency(self) -> Union[int, float]:
        return 1 / self.sampling_period

    @property
    def length_of_signal(self) -> int:
        return np.size(self.original_signal)

    def cal_naive_direct_fft(self) -> ndarray:
        """
        直接调用FFT函数的结果
        """
        return fft.fft(self.original_signal, self.n_fft)

    def _cal_single_sided_amplitude(self) -> ndarray:
        p2 = np.abs(self.cal_naive_direct_fft())
        if self.n_fft % 2 == 0:
            p1 = p2[:int(self.n_fft / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    def _cal_single_sided_angle(self) -> ndarray:
        a2 = np.angle(self.cal_naive_direct_fft())
        if self.n_fft % 2 == 0:
            a1 = a2[:int(self.n_fft / 2 + 1)]
            return a1
        else:
            raise Exception('TODO')

    def _cal_single_sided_frequency(self) -> ndarray:
        if self.n_fft % 2 == 0:
            return self.sampling_frequency * np.arange(0, self.n_fft / 2 + 1) / self.n_fft
        else:
            raise Exception('TODO')

    def single_sided_period_axis(self, period_unit: str = 'second') -> ndarray:
        """
        周期轴。根据fft (直接)计算出来的结果self.single_sided_frequency来推算
        """
        if self.n_fft % 2 == 0:
            return (1 / self._cal_single_sided_frequency()) / \
                   self.SupportedTransformedPeriod.get_by_convenient_period_unit_name(period_unit)
        else:
            raise Exception('TODO')

    def single_sided_frequency_axis(self, period_unit: str = 'second') -> ndarray:
        """
        频率轴。根据fft (直接)计算出来的结果self.single_sided_frequency来推算。
        或者可以很方便地根据self.single_sided_period_axis的结果来推算
        """
        single_sided_period_axis = self.single_sided_period_axis(period_unit)
        return 1 / single_sided_period_axis

    def single_sided_period_axis_all_supported(self, ordered_by_magnitude=False) -> pd.DataFrame:
        """
        集中所有信息的fft结果。周期向
        :param ordered_by_magnitude: 是否排序
        :return:
        """
        results = pd.DataFrame({'magnitude': self._cal_single_sided_amplitude(),
                                'log magnitude': np.log(self._cal_single_sided_amplitude()),
                                'phase angle (rad)': self._cal_single_sided_angle()})
        for value in self.SupportedTransformedPeriod:
            results[value.value[0]] = (1 / self._cal_single_sided_frequency()) / value.value[-1]
        results.index = self._cal_single_sided_frequency()
        results = results.rename_axis('frequency')
        if not ordered_by_magnitude:
            return results
        else:
            return results.sort_values(by=['magnitude'], ascending=False)

    def single_sided_frequency_axis_all_supported(self, ordered_by_magnitude=False) -> pd.DataFrame:
        """
        集中所有信息的fft结果。频率向
        :param ordered_by_magnitude: 是否排序
        :return:
        """
        results = pd.DataFrame({'magnitude': self._cal_single_sided_amplitude(),
                                'log magnitude': np.log(self._cal_single_sided_amplitude()),
                                'phase angle (rad)': self._cal_single_sided_angle()})
        for value in self.SupportedTransformedPeriod:
            results[value.value[1]] = self._cal_single_sided_frequency() * value.value[-1]
        results.index = self._cal_single_sided_frequency()
        results = results.rename_axis('frequency')
        if not ordered_by_magnitude:
            return results
        else:
            return results.sort_values(by=['magnitude'], ascending=False)

    def plot(self,
             considered_frequency_units: Union[str, Tuple[str, ...]] = None, *,
             overridden_plot_x_lim: Tuple[Union[int, float, None], Union[int, float, None]] = None,
             save_as_docx_path: Path = None) -> tuple:
        """
        画频谱图和相位图
        :return: 最后一组fft的frequency和phase的图的buf或者gca
        """
        return_f, return_p = None, None

        full_results_to_be_plot = self.single_sided_frequency_axis_all_supported()
        if isinstance(considered_frequency_units, str):
            considered_frequency_units = (considered_frequency_units,)
        considered_frequency_units = \
            considered_frequency_units or self.SupportedTransformedPeriod.list_all_convenient_frequency_unit_names()
        # %% 如果要存成docx，那就准备buffer
        if save_as_docx_path:
            # save_as_docx_buff.key就是图像的名字
            # save_as_docx_buff.value的形式是[buffer，宽度]
            sorted_key = np.array(
                [list(map(lambda x: self.name + ' ' + x + ' (magnitude)', considered_frequency_units)),
                 list(map(lambda x: self.name + ' ' + x + ' (phase angle)', considered_frequency_units))]).flatten('F')
            save_as_docx_buff = {key: [None, None] for key in sorted_key}
            save_to_buffer = True
        else:
            save_as_docx_buff = {}
            save_to_buffer = False

        def plot_single(_this_considered_frequency_unit,
                        x_lim=(None, None)):

            x = full_results_to_be_plot[_this_considered_frequency_unit].values
            # frequency
            buf_f = stem(x=x,
                         y=full_results_to_be_plot['magnitude'].values,
                         x_lim=x_lim,
                         infer_y_lim_according_to_x_lim=True,
                         x_label=f'Frequency ({_this_considered_frequency_unit})',
                         y_label='Magnitude',
                         save_to_buffer=False)
            if save_to_buffer:
                if not save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)'][0]:
                    save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)'][0] = buf_f
                else:
                    save_as_docx_buff.setdefault(self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)_2',
                                                 (buf_f, None))
            # phase
            buf_p = stem(x=x,
                         y=full_results_to_be_plot['phase angle (rad)'].values,
                         x_lim=x_lim,
                         x_label=f'Frequency ({_this_considered_frequency_unit})',
                         y_label='Phase angle (rad)',
                         save_to_buffer=save_to_buffer)
            if save_to_buffer:
                if not save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)'][0]:
                    save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)'][0] = buf_p
                else:
                    save_as_docx_buff.setdefault(self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)_2',
                                                 (buf_p, None))
            return buf_f, buf_p

        for this_considered_frequency_unit in considered_frequency_units:
            value = self.SupportedTransformedPeriod.get_by_convenient_frequency_unit_name(
                this_considered_frequency_unit
            )

            if overridden_plot_x_lim is None:
                try:
                    _ = value[2][0][0]
                    # 子1
                    plot_single(this_considered_frequency_unit,
                                value[2][0])
                    # 子2
                    return_f, return_p = plot_single(this_considered_frequency_unit,
                                                     value[2][1])
                except (IndexError, TypeError) as _:  # 说明'plot_x_lim'要么是()，要么是(x, y)
                    return_f, return_p = plot_single(this_considered_frequency_unit,
                                                     value[2])
            else:
                return_f, return_p = plot_single(this_considered_frequency_unit,
                                                 overridden_plot_x_lim)
        if save_to_buffer:
            put_cached_png_into_a_docx(save_as_docx_buff, save_as_docx_path, 2)
        return return_f, return_p

    def top_n_high(self):
        pass

    def find_peaks_of_fft_frequency(self):
        """
        找到频谱图中最高的local maximum
        """
        pass


class STFTProcessor(FFTProcessor):
    __slots__ = ('scipy_signal_stft_results', 'original_signal_as_time_series')

    def __init__(self, original_signal_as_time_series, *,
                 sampling_period: Union[int, float],
                 name: str):
        super().__init__(original_signal_as_time_series.values.flatten(),
                         sampling_period=sampling_period,
                         name=name)
        self.original_signal_as_time_series = original_signal_as_time_series

    def call_scipy_signal_stft(self, frequency_unit: str = None, time_axis_denominator: int = None, **kwargs):
        """
        scipy.signal.stft
        :param frequency_unit stft的频率轴的单位
        :param time_axis_denominator stft的时间轴的normalisation系数
        :return Tuple[ndarray频率, ndarray时间, ComplexNdarray傅里叶的复数结果]

        注意，frequency_unit和time_axis_denominator不应该耦合，它们可以独立调整。另外，虽然自动化程度不高，
        time_axis_denominator依然最好手动设置，因为stft直接出来的时间轴的scale就已经和self.sampling_frequency耦合了，
        如果再去做自动化的time_axis_denominator推测的话，代码会过于臃肿且复杂
        """
        if kwargs.get('nperseg') is None:
            raise Exception("'nperseg' should be given")
        scipy_signal_stft_results = list(stft(self.original_signal, fs=self.sampling_frequency, **kwargs))
        if frequency_unit is not None:
            scipy_signal_stft_results[0] *= self.SupportedTransformedPeriod.get_by_convenient_frequency_unit_name(
                frequency_unit
            )[-1]
        if time_axis_denominator is not None:
            scipy_signal_stft_results[1] /= time_axis_denominator
        return scipy_signal_stft_results

    def plot_scipy_signal_stft_aggregated(self, call_scipy_signal_stft_args: dict = None):
        """
        画scipy.signal.stft的结果
        """
        call_scipy_signal_stft_args = call_scipy_signal_stft_args or {}
        scipy_signal_stft_results = self.call_scipy_signal_stft(**call_scipy_signal_stft_args)
        pcolormesh(x=scipy_signal_stft_results[1],
                   y=scipy_signal_stft_results[0],
                   color_value=np.abs(scipy_signal_stft_results[2]),
                   y_lim=(0, 2.5))

        tt = np.abs(scipy_signal_stft_results[2])
        tt_angle = np.angle(scipy_signal_stft_results[2])
        ax = series(tt[2], label='day')
        ax = series(tt[4], ax=ax, label='half day')

        ax = series(tt_angle[2], label='day')
        ax = series(tt_angle[4], ax=ax, label='half day')

    def plot_scipy_signal_stft(self, window_length: datetime.timedelta, window='hann'):
        window_data = self.original_signal_as_time_series.to_windowed_time_series(
            window_length=window_length,
            window=window)  # type: WindowedTimeSeries

        # TEST
        for test_day in [7, 189]:
            data = window_data[test_day].values.flatten()
            this_fft_processor = FFTProcessor(
                data,
                sampling_period=window_data[test_day].adjacent_recordings_timedelta.seconds,
                name=f'Datetime = {window_data[test_day].first_valid_index()}to '
                     f'{window_data[test_day].last_valid_index()}',
                n_fft=window_data[test_day].__len__()*1024)
            f_plot, p_plot = this_fft_processor.plot('1/half day', overridden_plot_x_lim=(0.55, 1.55))
            f_plot = adjust_lim_label_ticks(f_plot, y_lim=(0, None))
            # series(data)

        for this_windows_data in window_data:
            this_windows_data_fft = FFTProcessor(
                this_windows_data.values.flatten(),
                sampling_period=this_windows_data.adjacent_recordings_timedelta.seconds,
                name=f'Datetime = {this_windows_data.first_valid_index()}to '
                     f'{this_windows_data.last_valid_index()}')

            tt = 1
        # this_day = self.original_signal_as_time_series.getitem_use_sliding_window(0,
        #                                                                           window_length=window_length,
        #                                                                           window=window)
        BufferedFigureSaver()
