import numpy as np
import pandas as pd
from numpy import ndarray, complex
from pandas import DataFrame
from scipy import fftpack
from datetime import timedelta
from typing import Union


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
    def fft_results(self) -> ndarray:
        return fftpack.fft(self.original_signal)

    @property
    def single_sided_amplitude_spectrum(self) -> ndarray:
        p2 = np.abs(self.fft_results)
        if self.length_of_signal % 2 == 0:
            p1 = p2[:int(self.length_of_signal / 2 + 1)]
            return p1
        else:
            raise Exception('TODO')

    @property
    def single_sided_angle(self) -> ndarray:
        a2 = np.angle(self.fft_results)
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
            return 1 / self.single_sided_frequency_axis
        else:
            raise Exception('TODO')
