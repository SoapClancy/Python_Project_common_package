import numpy as np
from numpy import ndarray
import copy
from Data_Preprocessing import float_eps


class OneDimMarkovChain:

    def __init__(self, *, current_state=None, next_state=None, **kwargs):
        # 第一步是去除nan
        if (current_state is not None) and (next_state is not None):
            not_nan_flag = np.bitwise_and(~np.isnan(current_state), ~np.isnan(next_state))
            self.unique_state = np.unique(current_state[~np.isnan(current_state)])  # type: ndarray
            self.current_state_and_next_state = np.vstack((copy.deepcopy(current_state[not_nan_flag]),
                                                           copy.deepcopy(next_state[not_nan_flag])))  # type: ndarray
        else:
            self.unique_state = self.current_state_and_next_state = None
        self.state_markov_chain_in_matrix = \
            kwargs.get('state_markov_chain_in_matrix') or self.cal_state_markov_chain_in_matrix()

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

    def sample_next_state_from_current_state(self, *, current_state):
        uniform_rnd = np.random.rand(1)
        row_idx = current_state == self.unique_state
        this_cdf = self.state_markov_chain_in_matrix['cdf'][row_idx, :]
        col_idx = np.argwhere(this_cdf.flatten() > uniform_rnd)[0]
        return self.unique_state[col_idx]

    def get_next_state_pdf_from_current_state(self, current_state):
        row_idx = current_state == self.unique_state
        this_pdf = self.state_markov_chain_in_matrix['pdf'][row_idx, :]
        return this_pdf
