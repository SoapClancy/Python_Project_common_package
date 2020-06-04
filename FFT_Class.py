import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fft
from datetime import timedelta
from typing import Union, Tuple, Callable, Any
from Ploting.fast_plot_Func import *
from enum import Enum, unique
from pathlib import Path
from Writting.utils import put_cached_png_into_a_docx
from scipy.signal import stft, find_peaks
from NdarraySubclassing import ComplexNdarray, OneDimensionNdarray
from Ploting.utils import BufferedFigureSaver
from TimeSeries_Class import WindowedTimeSeries, TimeSeries
from Ploting.adjust_Func import adjust_lim_label_ticks
from inspect import Parameter, Signature
import inspect
import math
from Data_Preprocessing.utils import scale_to_minus_plus_one
import copy
from sklearn.linear_model import Lasso
import warnings
from collections import namedtuple
from itertools import combinations
from Regression_Analysis.Models import BayesianRegression
import torch
import pyro.distributions as dist


USE_DEGREE_FOR_PLOT = True


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

    def cal_scipy_fft_fft(self) -> ndarray:
        """
        直接调用FFT函数的结果
        """
        return fft.fft(self.original_signal, self.n_fft)

    def _cal_single_sided_amplitude(self) -> ndarray:
        p2 = np.abs(self.cal_scipy_fft_fft())
        if self.n_fft % 2 == 0:
            p1 = p2[:int(self.n_fft / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    def _cal_single_sided_angle(self) -> ndarray:
        a2 = np.angle(self.cal_scipy_fft_fft())
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
                         ax=None,
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
                         y=full_results_to_be_plot['phase angle (rad)'].values if not USE_DEGREE_FOR_PLOT
                         else full_results_to_be_plot['phase angle (rad)'] * 180 / float(np.pi),
                         ax=None,
                         x_lim=x_lim,
                         x_label=f'Frequency ({_this_considered_frequency_unit})',
                         y_label=f"Phase angle ({'rad' if not USE_DEGREE_FOR_PLOT else 'degree'})",
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

    def find_peaks_of_fft_frequency(self, considered_frequency_unit: str, *,
                                    plot_args: dict = None,
                                    scipy_signal_find_peaks_args: dict = None,
                                    base_freq_is_a_peak: bool = True) -> tuple:
        """
        找到频谱图中最高的local maximum
        :param considered_frequency_unit: 考虑的频率的单位

        :param plot_args: 调用self.plot函数的args，一个字典。key代表除了considered_frequency_units之外的signature。
        如果是None的话代表不画图，只要不是None，哪怕是空字典也要画图。
        特殊的：
        'only_plot_peaks': bool 可以指定是否只画peaks
        'annotation_for_peak_f_axis_indices': Union[list, tuple] 指定标注哪些peak的indices，比如[0]代表标注第一个peak
        'annotation_y_offset_for_f': Union[list, tuple] 对应'annotation_for_peak_f_axis_indices'，freq
        'annotation_y_offset_for_p': Union[list, tuple] 对应'annotation_for_peak_f_axis_indices'，phase


        :param scipy_signal_find_peaks_args: scipy.signal.find_peaks函数的args。
        如果为None的话就默认是{'height': 0,'prominence': 1}

        :param base_freq_is_a_peak: 默认频谱图的base (0 Hz)分量默认是个peak。
        因为scipy.signal.find_peaks函数并不能找到这个，所以要手动设置！因为没有负频率，所以函数在0处不连续

        :return 一个tuple，
        元素0表示fft在peak处的信息array[0]是considered_frequency_unit轴，array[1]是幅值，array[2]是角度
        元素1代表peaks_index
        元素1是None或者是self.plot的第一个返回值
        元素2是None或者是self.plot的第二个返回值
        """
        scipy_signal_find_peaks_args = scipy_signal_find_peaks_args or {'height': 0, 'prominence': 1}
        fft_results = self.single_sided_frequency_axis_all_supported()
        fft_results = fft_results[[considered_frequency_unit, 'magnitude', 'phase angle (rad)']]
        peaks_index, _ = find_peaks(fft_results.values[:, 1], **scipy_signal_find_peaks_args)
        # 考虑基频是不是peak
        if base_freq_is_a_peak:
            peaks_index = np.concatenate(([0], peaks_index))
        peak_fft_results = fft_results.iloc[peaks_index, :]
        peak_fft_results_values = peak_fft_results.values
        # 画图
        if plot_args is not None:
            plot_args.update({'considered_frequency_units': considered_frequency_unit})
            # 是否只画peak而不画fft
            if plot_args.get('only_plot_peaks'):
                f_plot, p_plot = None, None
            else:
                passed_args = {key: val for key, val in plot_args.items()
                               if key in inspect.signature(self.plot).parameters}
                f_plot, p_plot = self.plot(**passed_args)
            f_plot = stem(x=peak_fft_results_values[:, 0],
                          y=peak_fft_results_values[:, 1],
                          ax=f_plot, color='r',
                          x_label=f'Frequency ({considered_frequency_unit})',
                          y_label='Magnitude'
                          )

            p_plot = stem(x=peak_fft_results_values[:, 0],
                          y=peak_fft_results_values[:, 2] if not USE_DEGREE_FOR_PLOT
                          else peak_fft_results_values[:, 2] * 180 / float(np.pi),
                          ax=p_plot, color='r',
                          x_label=f'Frequency ({considered_frequency_unit})',
                          y_label=f"Phase angle ({'rad' if not USE_DEGREE_FOR_PLOT else 'degree'})",
                          )
            # 指定标准哪些peaks
            if plot_args.get('annotation_for_peak_f_axis_indices'):
                annotation_y_offset_for_f = plot_args.get('annotation_y_offset_for_f')
                annotation_y_offset_for_p = plot_args.get('annotation_y_offset_for_p')
                for i, this_peak_idx in enumerate(plot_args.get('annotation_for_peak_f_axis_indices')):
                    f_plot.annotate(f'f = {peak_fft_results_values[this_peak_idx, 0]},\n'
                                    f'magnitude = {peak_fft_results_values[this_peak_idx, 1]}',
                                    xy=peak_fft_results_values[this_peak_idx, [0, 1]],
                                    xytext=(peak_fft_results_values[this_peak_idx, 0],
                                            peak_fft_results_values[this_peak_idx, 1] + annotation_y_offset_for_f[i]),
                                    arrowprops=dict(facecolor='black', arrowstyle="->"),
                                    )
                    p_plot.annotate(f'f = {peak_fft_results_values[this_peak_idx, 0]},\n'
                                    f'phase = {peak_fft_results_values[this_peak_idx, 2]}',
                                    xy=peak_fft_results_values[this_peak_idx, [0, 2]],
                                    xytext=(peak_fft_results_values[this_peak_idx, 0],
                                            peak_fft_results_values[this_peak_idx, 2] + annotation_y_offset_for_p[i]),
                                    arrowprops=dict(facecolor='black', arrowstyle="->"),
                                    )

            return peak_fft_results, peaks_index, f_plot, p_plot
        else:
            return peak_fft_results, peaks_index, None, None


class FourierSeriesProcessorMeta(type):

    def __new__(mcs, clsname, bases, clsdict):
        if clsname in ('APFormFourierSeriesProcessor', 'SCFormFourierSeriesProcessor'):
            sig = inspect.Signature([Parameter(this_arg, Parameter.KEYWORD_ONLY, default=None)
                                     for this_arg in list(clsdict['__slots__']) + list(bases[0].__slots__)])
            clsdict['__init__sig'] = sig
        clsobj = super().__new__(mcs, clsname, bases, clsdict)
        return clsobj

    def change_to_another_form(cls):
        pass


class FourierSeriesProcessor(metaclass=FourierSeriesProcessorMeta):
    """
    'frequency'的单位是Hz
    'x_value'是指时域的x轴的坐标，单位是秒
    """
    __slots__ = ('frequency',)

    def __init__(self, *args, **kwargs):
        bound = self.__getattribute__('__init__sig').bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            if val is not None:
                if val.ndim > 1:
                    raise ValueError("傅里叶级数的属性必须通过ndim==1的ndarray初始化")
                setattr(self, name, val)
        for this_attr in self.__getattribute__('__init__sig').parameters:
            if this_attr not in bound.arguments:
                setattr(self, this_attr, np.full(self.frequency.shape, 1.0))

    @staticmethod
    def cal_usable_x_value(x_value) -> ndarray:
        if isinstance(x_value, ndarray):
            return x_value
        elif isinstance(x_value, pd.DatetimeIndex):
            seconds_ndarray = np.array(list(map(lambda x: (x - x_value[0]).seconds, x_value)))
            return seconds_ndarray
        else:
            raise TypeError("Unsupported x_value type, should be pd.DatetimeIndex or ndarray")

    def _form_callable_component_funcs(self) -> Tuple[Callable, ...]:
        pass

    def __call__(self, x_value: Union[pd.DatetimeIndex, ndarray],
                 scale_to_minus_plus_one_flag=False, *,
                 return_raw=False):
        """
        返回对于给定x_value值的FourierSeriesProcessor对象的输出

        :param x_value 需要计算的x坐标

        :param scale_to_minus_plus_one_flag 结果是否要scale到正负1之间

        :param return_raw 是否返回各个频率分量的相应
        """
        x_value = self.cal_usable_x_value(x_value)
        callable_component_funcs = self._form_callable_component_funcs()
        results = []
        for this_func in callable_component_funcs:
            results.append(this_func(x_value))
        raw_results = np.array(results)
        results = np.sum(np.array(results), axis=0)
        if scale_to_minus_plus_one_flag:
            results = scale_to_minus_plus_one(results)
        if not return_raw:
            return results, None
        else:
            return results, raw_results.T


class APFormFourierSeriesProcessor(FourierSeriesProcessor):
    __slots__ = ('magnitude', 'phase')

    def _form_callable_component_funcs(self) -> Tuple[Callable, ...]:
        component_funcs = []
        for i in range(self.frequency.size):
            this_magnitude = float(self.magnitude[i])
            this_frequency = float(self.frequency[i])
            this_phase = float(self.phase[i])
            # 又是python copy的问题，
            # component_funcs.append(lambda x: this_magnitude * np.cos(2 * np.pi * this_frequency * x - this_phase))
            # 会使得component_funcs中每个函数的调用结果都一样。因为深层的函数内部的参数的指向一样
            # 解决方法：转成str类，作为source code，用的时候在用eval执行
            source_code = f"lambda x: {this_magnitude} * np.cos(2 * np.pi * {this_frequency} * x - {this_phase})"
            component_funcs.append(eval(source_code))
        return tuple(component_funcs)

    @classmethod
    def init_using_fft_found_peaks(cls,
                                   fft_found_peaks: Tuple[pd.DataFrame, ndarray, Union[None, Any], Union[None, Any]],
                                   considered_peaks_index: Union[list, tuple]):
        """
        用FFTProcessor对象的find_peaks_of_fft_frequency的结果去生成APFormFourierSeriesProcessor对象
        """
        frequency = fft_found_peaks[0].index[considered_peaks_index].values
        magnitude = fft_found_peaks[0]['magnitude'].values[considered_peaks_index]
        phase = fft_found_peaks[0]['phase angle (rad)'].values[considered_peaks_index]
        self = APFormFourierSeriesProcessor(frequency=frequency, magnitude=magnitude, phase=phase)
        return self


class SCFormFourierSeriesProcessor(FourierSeriesProcessor):
    __slots__ = ('coefficient_a', 'coefficient_b')

    def _form_callable_component_funcs(self) -> Tuple[Callable, ...]:
        component_funcs = []
        # 注意，这个形式的cos和sin是两个函数分别不同的参数，所以矩阵的行数是APFormFourierSeriesProcessor的两倍
        # 偶数列代表cos，奇数列代表sin
        for i in range(self.frequency.size):
            this_frequency = float(self.frequency[i])
            this_coefficient_a = float(self.coefficient_a[i])
            this_coefficient_b = float(self.coefficient_b[i])
            source_code = f"lambda x: {this_coefficient_a} * np.cos(2 * np.pi * {this_frequency} * x)"
            component_funcs.append(eval(source_code))  # 偶数列代表cos
            source_code = f"lambda x: {this_coefficient_b} * np.sin(2 * np.pi * {this_frequency} * x)"
            component_funcs.append(eval(source_code))  # 奇数列代表sin

        return tuple(component_funcs)

    def combination_of_frequency_selector(self, remove_base: bool = False, *, call__raw_results: ndarray) -> tuple:
        """
        将__call__的raw_results进行排列组合

        :param remove_base 不考虑base

        :param call__raw_results __call__的raw_results

        :return namedtuple -> 属性frequency_combination是一个tuple，属性partly_combination_reconstructed是对应的时域结果
        """
        selector_template = np.full((call__raw_results.shape[1], 1), 0)
        PartlyCall = namedtuple("PartlyCall", ("frequency_combination", "partly_combination_reconstructed"))
        self_frequency_slicer = list(range(0, self.frequency.size))
        final_results = []
        # 生成可能的排列组合
        # 至少两个元素，最多全部选择
        for this_number_of_components in range(2, self.frequency.size):
            # 在这个元素数量下的组合
            all_combinations_under_this_number = combinations(self_frequency_slicer, this_number_of_components)
            for this_combination in all_combinations_under_this_number:
                # 如果不考虑base
                if remove_base and (0 in this_combination):
                    continue
                this_selector = copy.deepcopy(selector_template)
                this_selector[np.array(list(map(lambda x: [x * 2, x * 2 + 1], this_combination))).flatten()] = 1
                final_results.append(
                    PartlyCall(
                        frequency_combination=tuple(self.frequency[list(this_combination)]),
                        partly_combination_reconstructed=np.matmul(call__raw_results, this_selector).flatten()
                    )
                )

        return tuple(final_results)


class AdvancedFFTProcessor:
    __slots__ = ('frequency', 'target')

    def __init__(self, frequency: ndarray, *, target: TimeSeries):
        self.frequency = OneDimensionNdarray(frequency)
        if not isinstance(target, TimeSeries):
            raise ValueError("'target' attribute in LASSOFFTProcessor object must be an instance of TimeSeries class")
        self.target = target

    def _form_x_matrix(self) -> ndarray:
        """
        生成SCFormFourierSeriesProcessor对象，默认coefficient全是1，调用__call__方法生成对应矩阵
        :return ：偶数列代表cos, 奇数列代表sin
        """
        if not np.isclose(np.min(self.frequency), 0, rtol=1.e-16, atol=1.e-16):
            warnings.warn("要用到Advanced FFT fitting，那么必须要包含base分量，已经自动添加", UserWarning)
        self.frequency = np.concatenate(([0], self.frequency))
        sc_form_fourier_series_processor = SCFormFourierSeriesProcessor(
            frequency=self.frequency
        )
        _, x_matrix = sc_form_fourier_series_processor(self.target.index, return_raw=True)
        return x_matrix


class LASSOFFTProcessor(AdvancedFFTProcessor):

    def do_lasso_fitting(self, *args, fit_intercept=False, **kwargs) \
            -> Tuple[Lasso, ndarray, SCFormFourierSeriesProcessor]:
        """
        利用sklearn中的lasso去fit傅里叶级数
        :param fit_intercept: 不允许fit截距，因为那说到底就是base分量的幅值，base分量没有的话会被自动添加
        :param
        :return: 元素0是fit好的Lasso对象，
        元素1是Lasso对象预测的值（即：模型输出），
        元素2是基于Lasso对象的coef属性生成SCFormFourierSeriesProcessor对象
        """
        x_matrix = self._form_x_matrix()
        sklearn_lasso_class_args = Signature.from_callable(Lasso).bind(*args, **kwargs)
        lasso_class = Lasso(fit_intercept=fit_intercept,
                            **sklearn_lasso_class_args.arguments)
        xx = x_matrix
        lasso_class.fit(X=xx, y=self.target.values.flatten())
        prediction = lasso_class.predict(xx)
        return lasso_class, prediction, SCFormFourierSeriesProcessor(frequency=self.frequency,
                                                                     coefficient_a=lasso_class.coef_[0::2],
                                                                     coefficient_b=lasso_class.coef_[1::2])


class BayesianFFTProcessor(AdvancedFFTProcessor):

    def do_bayesian_fitting(self, *args, fit_intercept=False, **kwargs) \
            -> Tuple[BayesianRegression, ndarray, SCFormFourierSeriesProcessor]:
        """
        利用bayesian regression去fit傅里叶级数
        :param fit_intercept: 不允许fit截距，因为那说到底就是base分量的幅值，base分量没有的话会被自动添加
        :param
        :return: 元素0是fit好的BayesianRegression对象，
        元素1是BayesianRegression对象预测的值（即：模型输出），
        元素2是基于BayesianRegression对象的coef属性生成SCFormFourierSeriesProcessor对象
        """
        x_matrix = torch.tensor(self._form_x_matrix(), dtype=torch.float)
        y = torch.tensor(self.target.values.flatten(), dtype=torch.float)
        # 用lasso估计作为初值
        initial_guess = BayesianRegression.lasso_results(x_matrix, y, False)[0]
        # 用Laplace分布作为prior
        weight_prior = dist.Laplace(torch.tensor(initial_guess, dtype=torch.float).reshape([1, x_matrix.shape[-1]]),
                                    3.).to_event(2)
        bayesian_regression = BayesianRegression(x_matrix.shape[-1], 1,
                                                 fit_intercept=False,
                                                 weight_prior=weight_prior)
        # mcmc
        mcmc_run_results = bayesian_regression.run_mcmc(x_matrix, y,
                                                        num_samples=300, warmup_steps=100)
        # DEBUG

        tt=1



class STFTProcessor(FFTProcessor):
    __slots__ = ('original_signal_as_time_series',
                 'windowed_data',
                 'stft_results')

    def __init__(self, original_signal_as_time_series, *,
                 sampling_period: Union[int, float],
                 name: str, **kwargs):
        super().__init__(original_signal_as_time_series.values.flatten(),
                         sampling_period=sampling_period,
                         name=name)
        self.original_signal_as_time_series = original_signal_as_time_series
        # 动态生成windowed_data
        sig = Signature.from_callable(self)
        __call__params = {key: val for key, val in kwargs.items() if key in sig.parameters}
        try:
            self.windowed_data = self(**__call__params)  # type: WindowedTimeSeries
        except TypeError:
            self.windowed_data = None
        self.stft_results = None

    def __call__(self, *, window_length: datetime.timedelta,
                 window=None):
        windowed_data = self.original_signal_as_time_series.to_windowed_time_series(
            window_length=window_length,
            window=window)  # type: WindowedTimeSeries
        self.windowed_data = windowed_data
        return windowed_data

    def do_stft(self, fft_processor_n_fft: int) -> Tuple[FFTProcessor, ...]:
        if self.windowed_data is None:
            raise Exception('No windowed data! Run __call__ first!')
        windowed_data_fft = np.empty((math.ceil(self.windowed_data.__len__()),), dtype=FFTProcessor)  # 也许用ndarray比[]快
        for i, this_windowed_data in enumerate(self.windowed_data):
            this_windows_data_fft = FFTProcessor(
                this_windowed_data.values.flatten(),
                sampling_period=this_windowed_data.adjacent_recordings_timedelta.seconds,
                name=f'Datetime = {this_windowed_data.first_valid_index()} to '
                     f'{this_windowed_data.last_valid_index()}',
                n_fft=fft_processor_n_fft)
            windowed_data_fft[i] = this_windows_data_fft
        self.stft_results = tuple(windowed_data_fft)
        return self.stft_results

    def find_peaks_of_fft_frequency(self, considered_frequency_unit: str, *,
                                    considered_window_index: Union[tuple, range] = (),
                                    plot_args: dict = None,
                                    scipy_signal_find_peaks_args: dict = None,
                                    base_freq_is_a_peak: bool = True):
        """
        :return: (!!!只有plot_args的时候才有数值return)
        第一维代表window的index，第二维度是第几个peak的频率分量，第三维度是(频率，幅值，角度)
        """
        sig = inspect.signature(FFTProcessor.find_peaks_of_fft_frequency)
        pass_args = {key: val for key, val in locals().items() if (key in sig.parameters and key != 'self')}
        if self.stft_results is None:
            raise Exception("Should call do_stft method at first")
        peaks_of_fft_results = np.empty((considered_window_index.__len__(),), dtype=FFTProcessor)  # 也许用ndarray比[]快
        for i, this_window_index in enumerate(considered_window_index):
            temp = self.stft_results[this_window_index].find_peaks_of_fft_frequency(**pass_args)
            peaks_of_fft_results[i] = temp
        # 提取数值结果
        numeric_only = []
        for i in peaks_of_fft_results:
            numeric_only.append(i[0].values)
        numeric_only = np.array(numeric_only)
        """
        DEBUG
        """
        tt = numeric_only[:, 1, 1]
        series(tt, title='1st peak magnitude change')
        tt = numeric_only[:, 1, 0]
        series(tt, title='1st peak frequency change')
        tt = numeric_only[:, 1, 2]
        series(tt, title='1st peak phase change')

        tt = numeric_only[:, 2, 1]
        series(tt, title='2nd peak magnitude change')
        tt = numeric_only[:, 2, 0]
        series(tt, title='2nd peak frequency change')
        tt = numeric_only[:, 2, 2]
        series(tt, title='2nd peak phase change')

        return numeric_only  # 第一维代表window的index，第二维度是第几个peak的频率分量，第三维度是(频率，幅值，角度)

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
