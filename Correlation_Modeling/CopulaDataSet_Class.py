from Ploting.fast_plot_Func import *
from PhysicalInstance_Class import PhysicalInstance
import pandas as pd
from pathlib import Path
from typing import Dict
from File_Management.load_save_Func import *
import tensorflow as tf
import tensorflow_probability as tfp
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray
import edward2 as ed
from sklearn.mixture import GaussianMixture
from UnivariateAnalysis_Class import *

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")


class VineCopulaDataset:
    def __init__(self, *, vine_copula_structure: tuple,
                 original_data: Union[PhysicalInstance, pd.DataFrame],
                 marginal_distribution_folder_path: Path,
                 marginal_distribution_name_dict: Dict[str, str]):
        # %% Assert 1: PhysicalInstance
        assert isinstance(original_data, PhysicalInstance)
        # %% Assert 2: no nan
        assert np.sum(np.nan(original_data.pd_view().values)) == 0
        # %% Assert 3: predictor dim number + dependant dim number is valid under vine_copula_structure
        self._assert_dim_num_valid(vine_copula_structure, original_data.columns.__len__())

        # %% load or fit marginal distribution obj
        self.original_data = original_data
        self.marginal_distribution = self.get_marginal_distribution(
            marginal_distribution_folder_path,
            marginal_distribution_name_dict
        )

    @staticmethod
    def _assert_dim_num_valid(vine_copula_structure: tuple, data_dim_num: int):
        def recursion(tuple_obj, now_unique):
            for ele in tuple_obj:
                if isinstance(ele, int):
                    now_unique.add(ele)
                else:
                    recursion(ele, now_unique)

        uniques = set()
        recursion(vine_copula_structure, uniques)
        assert max(uniques) == data_dim_num

    def get_marginal_distribution(self, marginal_distribution_folder_path: Path,
                                  marginal_distribution_name_dict: Dict[str, str]) -> Dict[str, object]:
        @load_exist_pkl_file_otherwise_run_and_save(
            marginal_distribution_folder_path / Path("marginal_distribution_meta.pkl"))
        def marginal_distribution_args():
            meta = {_key: None for _key in self.original_data.concerned_dim}
            for _key, _val in marginal_distribution_name_dict.items():
                if _val == "gmm":
                    gmm_model = [Univariate(self.original_data[_key]).fit_using_gaussian_mixture_model() for x in
                                 self.ndarray_data.T]

                elif _val == "mixture of truncated normal":
                    pass
                elif _val == "mixture of von mises":
                    pass
                elif _val == "mixture of logit normal":
                    pass
                else:
                    raise NotImplementedError
            return meta

        args = marginal_distribution_args()
        # Call constructor
        marginal_distribution = {key: None for key in self.original_data.concerned_dim}
        for key, val in marginal_distribution_name_dict.items():
            if val == "gmm":
                pass
            elif val == "mixture of truncated normal":
                pass
            elif val == "mixture of von mises":
                pass
            elif val == "mixture of logit normal":
                pass
            else:
                raise NotImplementedError
        return marginal_distribution
