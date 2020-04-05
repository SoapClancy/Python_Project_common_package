import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fftpack
from datetime import timedelta
from typing import Union
from Ploting.fast_plot_Func import series, hist, scatter


class FFTProcessor:
    __slots__ = ('original_signal', 'sampling_period')

    def __init__(self, original_signal: ndarray, sampling_period: int):
        if original_signal.ndim > 1:
            raise Exception('Only consider 1-D data')
        self.original_signal = original_signal
        self.sampling_period = sampling_period

    @property
    def sampling_frequency(self) -> Union[int, float]:
        return 1 / self.sampling_period

    @property
    def length_of_signal(self) -> int:
        return np.size(self.original_signal)

    @property
    def fft_results_direct(self) -> ndarray:
        return fftpack.fft(self.original_signal)

    @property
    def single_sided_amplitude_spectrum(self) -> ndarray:
        p2 = np.abs(self.fft_results_direct)
        if self.length_of_signal % 2 == 0:
            p1 = p2[:int(self.length_of_signal / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    @property
    def single_sided_angle(self) -> ndarray:
        a2 = np.angle(self.fft_results_direct)
        if self.length_of_signal % 2 == 0:
            a1 = a2[:int(self.length_of_signal / 2 + 1)]
            return a1
        else:
            raise Exception('TODO')

    @property
    def single_sided_frequency_axis(self) -> ndarray:
        if self.length_of_signal % 2 == 0:
            return self.sampling_frequency * np.arange(0, self.length_of_signal / 2 + 1) / self.length_of_signal
        else:
            raise Exception('TODO')

    def single_sided_period_axis(self, transform_to: str = 'second') -> ndarray:
        if self.length_of_signal % 2 == 0:
            if transform_to == 'hour':
                return (1 / self.single_sided_frequency_axis) / 3600
            elif transform_to == 'minute':
                return (1 / self.single_sided_frequency_axis) / 60
            elif transform_to == 'day':
                return (1 / self.single_sided_frequency_axis) / (60 * 60 * 24)
            elif transform_to == 'week':
                return (1 / self.single_sided_frequency_axis) / (60 * 60 * 24 * 7)
            elif transform_to == '28 days':
                return (1 / self.single_sided_frequency_axis) / (60 * 60 * 24 * 28)
            elif transform_to == 'year':
                return (1 / self.single_sided_frequency_axis) / (60 * 60 * 24 * 365)
            return 1 / self.single_sided_frequency_axis
        else:
            raise Exception('TODO')

    def single_sided_fft_results_usable(self, ordered_by_magnitude=False) -> pd.DataFrame:
        """
        集中所有信息的fft结果
        :param ordered_by_magnitude: 是否排序
        :return:
        """
        results = pd.DataFrame({'magnitude': self.single_sided_amplitude_spectrum,
                                'log magnitude': np.log(self.single_sided_amplitude_spectrum),
                                'phase angle (rad)': self.single_sided_angle,
                                'period (second)': self.single_sided_period_axis(),
                                'period (minute)': self.single_sided_period_axis('minute'),
                                'period (hour)': self.single_sided_period_axis('hour'),
                                'period (day)': self.single_sided_period_axis('day'),
                                'period (week)': self.single_sided_period_axis('week'),
                                'period (28 days)': self.single_sided_period_axis('28 days'),
                                'period (year)': self.single_sided_period_axis('year')},
                               index=[self.single_sided_frequency_axis]).rename_axis('frequency')  # type: DataFrame
        if not ordered_by_magnitude:
            return results
        else:
            return results.sort_values(by=['magnitude'], ascending=False)

    def top_n_high(self):
        pass
