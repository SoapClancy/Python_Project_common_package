import numpy as np
from numpy import ndarray
from Data_Preprocessing import float_eps
from typing import Union, Callable, Sequence
from ConvenientDataType import OneDimensionNdarray
from BivariateAnalysis_Class import MethodOfBins
import tensorflow_probability as tfp
import warnings
from scipy.interpolate import interp1d


class OneDimMarkovChain:
    """
    "raw (data)": Union[int, float] the data in the raw measurement domain
    "digitize (data)": Union[int, float] the bin data of raw measurement
    "state (data)": Union[int] the data that can are actually known/used by the OneDimMarkovChain obj.
     If the obj is initialised by using init_from_one_dim_ndarray, then state is also the indices of digitize bins
    """

    def __init__(self, *, current_state: ndarray = None, next_state: ndarray = None,
                 state_markov_chain_in_matrix: dict = None,
                 raw_to_state_func: Callable = None, state_to_digitize_func: Callable = None,
                 use_tensorflow_probability: bool = True):
        build_new_mc_bool = (current_state is not None) and (next_state is not None)
        use_existing_mc_bool = all((current_state is None, next_state is None,
                                    state_markov_chain_in_matrix is not None))
        # Should either build new MC or use the existing one, not both
        assert (build_new_mc_bool or use_existing_mc_bool) and not (build_new_mc_bool and use_existing_mc_bool)

        self.use_tensorflow_probability = use_tensorflow_probability
        if not use_tensorflow_probability:
            warnings.warn("TensorFlow Probability (TFP) function will not be used, which may result in significant "
                          "performance issues! Please consider to use TFP")
        if build_new_mc_bool:
            mask = np.bitwise_and(~np.isnan(current_state), ~np.isnan(next_state))
            self.unique_state = np.sort(np.unique(np.concatenate((current_state[mask], next_state[mask]))))
            self.current_state_and_next_state = np.stack((current_state[mask], next_state[mask]), axis=1)
            self.state_markov_chain_in_matrix = self.cal_state_markov_chain_in_matrix()
        else:
            self.unique_state = state_markov_chain_in_matrix["name"]
            self.state_markov_chain_in_matrix = state_markov_chain_in_matrix

        self.raw_to_state_func = raw_to_state_func
        self.state_to_digitize_func = state_to_digitize_func

    @classmethod
    def init_from_one_dim_ndarray(cls, one_dim_ndarray: ndarray, resolution: Union[float, int]):
        one_dim_ndarray = OneDimensionNdarray(one_dim_ndarray)
        one_dim_ndarray = one_dim_ndarray[~np.isnan(one_dim_ndarray)]
        _min = np.min(one_dim_ndarray).item()
        _max = np.max(one_dim_ndarray).item()
        bins = MethodOfBins.cal_array_of_bin_boundary(_min, _max + resolution, resolution)
        bins = bins[:, 0]
        state = np.digitize(one_dim_ndarray, bins).astype(int)

        def raw_to_state_func(raw_data: Union[ndarray, int, float]):
            return np.digitize(raw_data, bins).astype(int)

        def state_to_digitize_func(_state: Union[ndarray, int, float]):
            def python_func(x):
                _bin_left = bins[x - 1]
                _bin_right = bins[x]
                return _bin_left, (_bin_left + _bin_right) / 2, _bin_right

            return np.stack(np.frompyfunc(python_func, 1, 1)(_state))

        self = cls(current_state=state[:-1], next_state=state[1:],
                   raw_to_state_func=raw_to_state_func, state_to_digitize_func=state_to_digitize_func)
        return self

    def cal_state_markov_chain_in_matrix(self) -> dict:
        state_markov_chain = np.full((self.unique_state.size, self.unique_state.size), 0)
        for row_idx, current_state in enumerate(self.unique_state):
            for col_idx, next_state in enumerate(self.unique_state):
                # Count the number of transitions
                mask_current = self.current_state_and_next_state[:, 0] == current_state
                mask_next = self.current_state_and_next_state[:, 1] == next_state
                number_of_transition = np.sum(np.bitwise_and(mask_current, mask_next))
                state_markov_chain[row_idx, col_idx] = number_of_transition
        state_markov_chain_pmf = self.__cal_pmf(state_markov_chain=state_markov_chain)
        state_markov_chain_cmf = self.__cal_cmf(state_markov_chain_pmf=state_markov_chain_pmf)
        return {
            'numbers': state_markov_chain,  # type: ndarray
            'pmf': state_markov_chain_pmf,  # type: ndarray
            'cmf': state_markov_chain_cmf,  # type: ndarray
            'name': self.unique_state  # type:ndarray
        }

    @staticmethod
    def __cal_pmf(*, state_markov_chain: ndarray):
        sum_on_row = np.sum(state_markov_chain, axis=1).reshape(-1, 1)
        sum_on_row = np.tile(sum_on_row, (1, state_markov_chain.shape[1]))
        state_markov_chain_pmf = state_markov_chain / sum_on_row
        return state_markov_chain_pmf

    @staticmethod
    def __cal_cmf(*, state_markov_chain_pmf: ndarray):
        cmf = np.cumsum(state_markov_chain_pmf, axis=1)
        max_on_row = cmf[:, -1].reshape(-1, 1)
        max_on_row = np.tile(max_on_row, (1, cmf.shape[1]))
        cmf = cmf / max_on_row * (1 - float_eps)
        return cmf

    def sample_the_next_from_current_state(self, *, current_state, number_of_samples: int = 1,
                                           output_digitize: bool, resample: bool):
        """
        :param current_state
        :param number_of_samples
        :param output_digitize: Output the next state as "state (data)" or "digitize (data)". Since "state (data)" is
        even not in the measurement domain, it is better to set output_state to False and output as "digitize (data)".
        :param resample: (Only valid if output_digitize is True) Resample the values in each bins (assume uniform).
        """
        output_domain = self.state_to_digitize_func(self.unique_state) if output_digitize else self.unique_state
        if self.use_tensorflow_probability:
            pmf = self.get_next_state_pmf_from_current_state(current_state)
            dist = tfp.distributions.__getattribute__("Categorical")(probs=pmf)
            col_idx = dist.sample(number_of_samples).numpy()
            results = output_domain[col_idx]
        else:
            uniform_rnd = np.random.rand(number_of_samples)
            cmf = self.get_next_state_cmf_from_current_state(current_state)
            col_idx = [np.argwhere(cmf > now)[0].item() for now in uniform_rnd]
            results = output_domain[col_idx]
        if output_digitize and resample:
            results = np.random.uniform(results[:, 0], results[:, 1])
        return results

    def get_next_state_pmf_from_current_state(self, current_state):
        row_idx = current_state == self.unique_state
        pmf = self.state_markov_chain_in_matrix['pmf'][row_idx, :]
        return pmf.flatten()

    def get_next_state_cmf_from_current_state(self, current_state):
        row_idx = current_state == self.unique_state
        cmf = self.state_markov_chain_in_matrix['cmf'][row_idx, :]
        return cmf.flatten()

    def get_next_digitize_range_from_current_raw(self, current_raw,
                                                 percentiles: Sequence[Union[int, float]], *,
                                                 method: str = "interpolation", resample: bool):
        assert method in ("interpolation", "sampling")
        # transfer raw to state
        current_state = self.raw_to_state_func(current_raw)

        if method == "interpolation":
            cmf = self.get_next_state_cmf_from_current_state(current_state)
            interp = interp1d(cmf, self.state_to_digitize_func(self.unique_state)[:, 1])
            results = interp(np.array(percentiles) / 100)
        else:
            samples = self.sample_the_next_from_current_state(current_state=current_state, number_of_samples=1_000_000,
                                                              output_digitize=True, resample=resample)
            results = np.percentile(samples, percentiles)
        return results

    def get_next_digitize_mean_from_current_raw(self, current_raw, *, method: str = "weighted average", resample: bool):
        assert method in ("weighted average", "sampling")
        # transfer raw to state
        current_state = self.raw_to_state_func(current_raw)

        if method == "weighted average":
            pmf = self.get_next_state_pmf_from_current_state(current_state)
            results = np.average(self.state_to_digitize_func(self.unique_state)[:, 1], weights=pmf)
        else:
            samples = self.sample_the_next_from_current_state(current_state=current_state, number_of_samples=1_000_000,
                                                              output_digitize=True, resample=resample)
            results = np.mean(samples)
        return results
