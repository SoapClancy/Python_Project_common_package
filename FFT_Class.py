import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fftpack
from datetime import timedelta
from typing import Union, Tuple
from Ploting.fast_plot_Func import series, hist, scatter, stem
from enum import Enum, unique


class FFTProcessor:
    __slots__ = ('original_signal', 'sampling_period', 'name')

    @unique
    class SupportedTransformedPeriod(Enum):
        # 命名方式是：
        # ('convenient_period_unit_name',
        # 'convenient_frequency_unit_name',
        # 'plot_x_lim', 👉 ()代表不设置限制，(x, y)代表一个图范围是x到y，((x1,y1), (x2,y2))代表两个图
        # 'scale_factor')
        second = ('second', '1/second', (0, None), 1)
        minute = ('minute', '1/minute', (-10e-4, None), 60)
        hour = ('hour', '1/hour', (-10e-4, None), 60 * 60)
        half_day = ('half day', '1/half day', (-0.05, None), 60 * 60 * 12)
        day = ('day', '1/day', (-0.1, 12.5), 60 * 60 * 24)
        week = ('week', '1/week', (3.5, 10.5), 60 * 60 * 24 * 7)
        _28_days = ('28 days', '1/28 days', (-0.25, None), 60 * 60 * 24 * 28)
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
        def all_convenient_period_unit_names(cls):
            return tuple([x.value[0] for x in cls])

        @classmethod
        def all_convenient_frequency_unit_names(cls):
            return tuple([x.value[1] for x in cls])

    def __init__(self, original_signal: ndarray, *, sampling_period: int, name: str = None):
        if original_signal.ndim > 1:
            raise Exception('Only consider 1-D data')
        self.original_signal = original_signal
        self.sampling_period = sampling_period
        self.name = name

    def __str__(self):
        return self.name

    @property
    def sampling_frequency(self) -> Union[int, float]:
        return 1 / self.sampling_period

    @property
    def length_of_signal(self) -> int:
        return np.size(self.original_signal)

    @property
    def _fft_results_direct(self) -> ndarray:
        return fftpack.fft(self.original_signal)

    @property
    def single_sided_amplitude(self) -> ndarray:
        p2 = np.abs(self._fft_results_direct)
        if self.length_of_signal % 2 == 0:
            p1 = p2[:int(self.length_of_signal / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    @property
    def single_sided_angle(self) -> ndarray:
        a2 = np.angle(self._fft_results_direct)
        if self.length_of_signal % 2 == 0:
            a1 = a2[:int(self.length_of_signal / 2 + 1)]
            return a1
        else:
            raise Exception('TODO')

    @property
    def single_sided_frequency(self) -> ndarray:
        if self.length_of_signal % 2 == 0:
            return self.sampling_frequency * np.arange(0, self.length_of_signal / 2 + 1) / self.length_of_signal
        else:
            raise Exception('TODO')

    def single_sided_period_axis(self, period_unit: str = 'second') -> ndarray:
        """
        周期轴。根据fft (直接)计算出来的结果self.single_sided_frequency来推算
        """
        if self.length_of_signal % 2 == 0:
            return (1 / self.single_sided_frequency) / \
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
        results = pd.DataFrame({'magnitude': self.single_sided_amplitude,
                                'log magnitude': np.log(self.single_sided_amplitude),
                                'phase angle (rad)': self.single_sided_angle})
        for value in self.SupportedTransformedPeriod:
            results[value.value[0]] = (1 / self.single_sided_frequency) / value.value[-1]
        results.index = self.single_sided_frequency
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
        results = pd.DataFrame({'magnitude': self.single_sided_amplitude,
                                'log magnitude': np.log(self.single_sided_amplitude),
                                'phase angle (rad)': self.single_sided_angle})
        for value in self.SupportedTransformedPeriod:
            results[value.value[1]] = self.single_sided_frequency * value.value[-1]
        results.index = self.single_sided_frequency
        results = results.rename_axis('frequency')
        if not ordered_by_magnitude:
            return results
        else:
            return results.sort_values(by=['magnitude'], ascending=False)

    def plot(self, considered_frequency_units: Tuple[str, ...] = None):
        """
        画频谱图和相位图
        """
        full_results_to_be_plot = self.single_sided_frequency_axis_all_supported()
        if considered_frequency_units is None:
            considered_frequency_units = self.SupportedTransformedPeriod.all_convenient_frequency_unit_names()

        def plot_single(_this_considered_frequency_unit,
                        _x_lim,
                        _y_lim_freq=None,
                        _y_lim_phase=None):
            stem(x=full_results_to_be_plot[_this_considered_frequency_unit].values,
                 y=full_results_to_be_plot['magnitude'].values,
                 x_lim=_x_lim,
                 y_lim=_y_lim_freq,
                 x_label=f'Frequency ({_this_considered_frequency_unit})',
                 y_label='Magnitude')
            stem(x=full_results_to_be_plot[_this_considered_frequency_unit].values,
                 y=full_results_to_be_plot['phase angle (rad)'].values,
                 x_lim=_x_lim,
                 y_lim=_y_lim_phase,
                 x_label=f'Frequency ({_this_considered_frequency_unit})',
                 y_label='Phase angle (rad)')

        for this_considered_frequency_unit in considered_frequency_units:
            value = self.SupportedTransformedPeriod.get_by_convenient_frequency_unit_name(
                this_considered_frequency_unit
            )
            try:
                x_lim = value[2][0][0]
                # 频率_子1
                plot_single(this_considered_frequency_unit,
                            value[2][0],
                            (-0.5, full_results_to_be_plot['magnitude'].values[:1000].max() * 1.3))
                # 频率_子2
                plot_single(this_considered_frequency_unit,
                            value[2][1],
                            (-0.5, None))
            except (IndexError, TypeError):  # 说明'plot_x_lim'要么是()，要么是(x, y)
                plot_single(this_considered_frequency_unit,
                            value[2],
                            (-0.5, None))

    def top_n_high(self):
        pass


class LASSOFFT(FFTProcessor):
    pass


class STFT(FFTProcessor):
    pass
