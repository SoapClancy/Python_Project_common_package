import numpy as np
from FFT_Class import FFTProcessor
from TimeSeries_Class import TimeSeries, merge_two_time_series_df
import pandas as pd
from Ploting.fast_plot_Func import *
from numpy import ndarray
from typing import Tuple, Callable


class FFTCorrelation:
    __slots__ = (
        'time_series',
        'correlation_func')


class BivariateFFTCorrelation(FFTCorrelation):

    def __init__(self, *,
                 main_time_series_df: pd.DataFrame,
                 vice_time_series_df: pd.DataFrame,
                 correlation_func: Tuple[Callable, ...]):
        self.time_series = merge_two_time_series_df(main_time_series_df,
                                                    vice_time_series_df)
        self.correlation_coefficient = correlation_func
        tt = 1

    def corr_between_pairwise_peaks_f(self, *,
                                      considered_frequency_unit: str,
                                      peak_f_axis_indices_for_main: tuple,
                                      peak_f_axis_indices_for_vice: tuple) -> ndarray:
        pass

    def corr_between_main_peaks_f_and_vice(self, *,
                                           considered_frequency_unit: str,
                                           peak_f_axis_indices_for_main: tuple) -> ndarray:
        pass

    def corr_between_combined_main_peaks_f_and_vice(self, *,
                                                    considered_frequency_unit: str,
                                                    peak_f_axis_indices_for_main: tuple
                                                    ) -> ndarray:
        pass
