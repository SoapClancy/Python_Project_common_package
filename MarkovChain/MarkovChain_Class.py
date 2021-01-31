import numpy as np
from numpy import ndarray
import copy
from Data_Preprocessing import float_eps
from typing import Union
from Ploting.fast_plot_Func import *
import pandas as pd
from ConvenientDataType import OneDimensionNdarray
from BivariateAnalysis_Class import MethodOfBins
from typing import Callable
import ctypes


class OneDimMarkovChain:

    def __init__(self, *, current_state=None, next_state=None,
                 raw_to_state_func: Callable = None, reverse_func: Callable = None, **kwargs):
        # Remove NaN
        if (current_state is not None) and (next_state is not None):
            not_nan_flag = np.bitwise_and(~np.isnan(current_state), ~np.isnan(next_state))
            self.unique_state = np.unique(
                np.concatenate((current_state[not_nan_flag], next_state[not_nan_flag]))
            )  # type: ndarray
            self.current_state_and_next_state = np.vstack((copy.deepcopy(current_state[not_nan_flag]),
                                                           copy.deepcopy(next_state[not_nan_flag])))  # type: ndarray
        else:
            self.unique_state = self.current_state_and_next_state = None
        self.state_markov_chain_in_matrix = \
            kwargs.get('state_markov_chain_in_matrix') or self.cal_state_markov_chain_in_matrix()
        self.raw_to_state_func = raw_to_state_func
        self.reverse_func = reverse_func

    @classmethod
    def init_from_one_dim_ndarray(cls, one_dim_ndarray: ndarray, resolution: Union[float, int]):
        one_dim_ndarray = OneDimensionNdarray(one_dim_ndarray)
        _min = np.nanmin(one_dim_ndarray).item()
        _max = np.nanmax(one_dim_ndarray).item()
        bins = MethodOfBins.cal_array_of_bin_boundary(_min, _max, resolution)
        bins = bins[:, 0]
        digit = np.digitize(one_dim_ndarray, bins).astype(int)

        def raw_to_state_func(raw_data: Union[ndarray, int, float]):
            return np.digitize(raw_data, bins).astype(int)

        def reverse_func(_digit: int):
            _bin_left = bins[_digit - 1]
            _bin_right = bins[_digit]
            return _bin_left, (_bin_left + _bin_right) / 2, _bin_right

        self = cls(current_state=digit[:-1], next_state=digit[1:],
                   raw_to_state_func=raw_to_state_func, reverse_func=reverse_func)
        return self

    def cal_state_markov_chain_in_matrix(self) -> dict:
        """
        计算这个Markov chain的转换矩阵
        :return: 转换矩阵的numbers, pdf，cdf和每行每列对应的名字（row_and_col_name）
        """
        state_markov_chain = np.full((self.unique_state.size, self.unique_state.size), 0)
        for row_idx, current_state in enumerate(self.unique_state):
            for col_idx, next_state in enumerate(self.unique_state):
                number_of_transition = self.__find_number_of_transition(current_state=current_state,
                                                                        next_state=next_state)
                state_markov_chain[row_idx, col_idx] = number_of_transition
        state_markov_chain_pdf = self.__cal_pdf(state_markov_chain=state_markov_chain)
        state_markov_chain_cdf = self.__cal_cdf(state_markov_chain_pdf=state_markov_chain_pdf)
        return {'numbers': state_markov_chain,
                'pdf': state_markov_chain_pdf,
                'cdf': state_markov_chain_cdf,
                'name': self.unique_state}

    def __find_number_of_transition(self, *, current_state, next_state):
        condition_on_current_state_flag = self.current_state_and_next_state[0, :] == current_state
        condition_on_next_state_flag = self.current_state_and_next_state[1, :] == next_state
        condition_on_current_and_next_state_flag = np.bitwise_and(condition_on_current_state_flag,
                                                                  condition_on_next_state_flag)
        return sum(condition_on_current_and_next_state_flag)

    @staticmethod
    def __cal_pdf(*, state_markov_chain: ndarray):
        sum_on_row = np.sum(state_markov_chain, axis=1).reshape(-1, 1)
        sum_on_row = np.tile(sum_on_row, (1, state_markov_chain.shape[1]))
        state_markov_chain_pdf = state_markov_chain / sum_on_row
        return state_markov_chain_pdf

    @staticmethod
    def __cal_cdf(*, state_markov_chain_pdf: ndarray):
        cdf = np.cumsum(state_markov_chain_pdf, axis=1)
        max_on_row = cdf[:, -1].reshape(-1, 1)
        max_on_row = np.tile(max_on_row, (1, cdf.shape[1]))
        cdf = cdf / max_on_row * (1 - float_eps)
        return cdf

    def sample_next_state_from_current_state(self, *, current_state, num: int = 1):
        uniform_rnd = np.random.rand(num)
        row_idx = current_state == self.unique_state
        this_cdf = self.state_markov_chain_in_matrix['cdf'][row_idx, :]
        if num == 1:
            col_idx = np.argwhere(this_cdf.flatten() > uniform_rnd)[0]
            return self.unique_state[col_idx]
        else:
            col_idx = [np.argwhere(this_cdf.flatten() > now)[0].item() for now in uniform_rnd]
            states = [self.unique_state[now] for now in col_idx]
            return states

    def get_next_state_pdf_from_current_state(self, current_state):
        row_idx = current_state == self.unique_state
        this_pdf = self.state_markov_chain_in_matrix['pdf'][row_idx, :]
        return this_pdf

    def get_next_state_cdf_from_current_state(self, current_state):
        row_idx = current_state == self.unique_state
        this_cdf = self.state_markov_chain_in_matrix['cdf'][row_idx, :]
        return this_cdf

    def get_next_digitize_range_from_current_raw(self, current_raw,
                                                 pct: Sequence):
        # to state
        current_state = self.raw_to_state_func(current_raw)
        this_cdf = self.get_next_state_cdf_from_current_state(current_state)
        pct_y = np.percentile(this_cdf.flatten(), pct)
        # the lower, the higher
        lower = 0
        higher = len(this_cdf.flatten()) - 1
        lower_diff_sign = np.sign(this_cdf - pct_y[0]).flatten()
        higher_diff_sign = np.sign(this_cdf - pct_y[1]).flatten()
        for i in range(1, len(this_cdf.flatten())):
            if lower_diff_sign[i] != lower_diff_sign[i - 1]:
                lower = i
            if higher_diff_sign[i] != higher_diff_sign[i - 1]:
                higher = i
        lower_state = self.unique_state[lower]
        higher_state = self.unique_state[higher]
        lower_digitize = self.reverse_func(lower_state)[1]
        higher_digitize = self.reverse_func(higher_state)[1]
        return [lower_digitize, higher_digitize]

    def get_next_digitize_mean_from_current_raw(self, current_raw):
        # to state
        current_state = self.raw_to_state_func(current_raw)
        samples = self.sample_next_state_from_current_state(current_state=current_state, num=10_000)
        next_digitize = [self.reverse_func(now)[1] for now in samples]
        return np.mean(next_digitize)
