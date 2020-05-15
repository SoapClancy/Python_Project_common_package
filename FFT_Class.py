import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fft
from datetime import timedelta
from typing import Union, Tuple
from Ploting.fast_plot_Func import series, hist, scatter, stem
from enum import Enum, unique
from pathlib import Path
from Writting.utils import put_cached_png_into_a_docx
from sklearn.linear_model import Lasso


class FFTProcessor:
    __slots__ = ('original_signal', 'sampling_period', 'name')

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

    def __init__(self, original_signal: ndarray, *, sampling_period: Union[int, float], name: str):
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

    def cal_naive_direct_fft(self) -> ndarray:
        """
        直接调用FFT函数的结果
        """
        return fft.fft(self.original_signal)

    def _cal_single_sided_amplitude(self) -> ndarray:
        p2 = np.abs(self.cal_naive_direct_fft())
        if self.length_of_signal % 2 == 0:
            p1 = p2[:int(self.length_of_signal / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    def _cal_single_sided_angle(self) -> ndarray:
        a2 = np.angle(self.cal_naive_direct_fft())
        if self.length_of_signal % 2 == 0:
            a1 = a2[:int(self.length_of_signal / 2 + 1)]
            return a1
        else:
            raise Exception('TODO')

    def _cal_single_sided_frequency(self) -> ndarray:
        if self.length_of_signal % 2 == 0:
            return self.sampling_frequency * np.arange(0, self.length_of_signal / 2 + 1) / self.length_of_signal
        else:
            raise Exception('TODO')

    def single_sided_period_axis(self, period_unit: str = 'second') -> ndarray:
        """
        周期轴。根据fft (直接)计算出来的结果self.single_sided_frequency来推算
        """
        if self.length_of_signal % 2 == 0:
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
             considered_frequency_units: Tuple[str, ...] = None, *,
             save_as_docx_path: Path = None):
        """
        画频谱图和相位图
        """
        full_results_to_be_plot = self.single_sided_frequency_axis_all_supported()
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
            buf = stem(x=x,
                       y=full_results_to_be_plot['magnitude'].values,
                       x_lim=x_lim,
                       infer_y_lim_according_to_x_lim=True,
                       x_label=f'Frequency ({_this_considered_frequency_unit})',
                       y_label='Magnitude',
                       save_to_buffer=False)
            if save_to_buffer:
                if not save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)'][0]:
                    save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)'][0] = buf
                else:
                    save_as_docx_buff.setdefault(self.name + ' ' + _this_considered_frequency_unit + ' (magnitude)_2',
                                                 (buf, None))
            # phase
            buf = stem(x=x,
                       y=full_results_to_be_plot['phase angle (rad)'].values,
                       x_lim=x_lim,
                       x_label=f'Frequency ({_this_considered_frequency_unit})',
                       y_label='Phase angle (rad)',
                       save_to_buffer=save_to_buffer)
            if save_to_buffer:
                if not save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)'][0]:
                    save_as_docx_buff[self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)'][0] = buf
                else:
                    save_as_docx_buff.setdefault(self.name + ' ' + _this_considered_frequency_unit + ' (phase angle)_2',
                                                 (buf, None))

        for this_considered_frequency_unit in considered_frequency_units:
            value = self.SupportedTransformedPeriod.get_by_convenient_frequency_unit_name(
                this_considered_frequency_unit
            )
            try:
                _ = value[2][0][0]
                # 子1
                plot_single(this_considered_frequency_unit,
                            value[2][0])
                # 子2
                plot_single(this_considered_frequency_unit,
                            value[2][1])
            except (IndexError, TypeError) as _:  # 说明'plot_x_lim'要么是()，要么是(x, y)
                plot_single(this_considered_frequency_unit,
                            value[2])
        if save_to_buffer:
            put_cached_png_into_a_docx(save_as_docx_buff, save_as_docx_path, 2)

    def top_n_high(self):
        pass


class STFT(FFTProcessor):
    pass
