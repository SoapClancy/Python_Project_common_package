from numpy import ndarray
from sklearn.cluster import KMeans
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from Ploting.fast_plot_Func import scatter
from typing import Tuple
import numpy as np
from BivariateAnalysis_Class import MethodOfBins
from ConvenientDataType import OneDimensionNdarray


class OneDimensionBinnedData:
    __slots__ = ("data", "bin")

    def __init__(self, data: OneDimensionNdarray,
                 *, bin_step: float,
                 first_bin_left_boundary: float = None,
                 last_bin_left_boundary: float = None, ):
        self.data = OneDimensionNdarray(data)
        if first_bin_left_boundary is None:
            first_bin_left_boundary = np.nanmin(data) - bin_step / 2
        if last_bin_left_boundary is None:
            last_bin_left_boundary = np.nanmax(data) - bin_step / 2
        self.bin = MethodOfBins.cal_array_of_bin_boundary(first_bin_left_boundary, last_bin_left_boundary, bin_step)

    def __call__(self, query_key: float) -> ndarray:
        """
        This function return which bin the given query_key locate in
        """
        larger_or_eq_mask = query_key >= self.bin[:, 0]
        smaller_mask = query_key < self.bin[:, 2]
        return self.bin[np.bitwise_and(larger_or_eq_mask, smaller_mask)][0]


class HighDimensionalAnalysis:
    __slots__ = ('ndarray_data', 'kmeans_file_')
    """
    默认ndarray_data的每一列代表一个维度，每一行代表一次记录
    """

    def __init__(self, ndarray_data: ndarray, kmeans_file_: str = None):
        if ndarray_data.ndim == 1:
            ndarray_data = ndarray_data.reshape(-1, 1)
        self.ndarray_data = ndarray_data  # type: ndarray
        self.kmeans_file_ = kmeans_file_  # type: str

    def dimension_reduction_by_kmeans(self, n_clusters: int = 5, **kmeans_args):
        @load_exist_pkl_file_otherwise_run_and_save(self.kmeans_file_)
        def fit_kmeans():
            nonlocal n_clusters
            n_clusters = kmeans_args.pop('n_clusters') if 'n_clusters' in kmeans_args else n_clusters
            kmeans = KMeans(n_clusters=n_clusters, **kmeans_args)
            return kmeans.fit(self.ndarray_data)

        kmeans_model = fit_kmeans  # type: KMeans

        return kmeans_model.predict(self.ndarray_data)

    def hierarchical_data_split(self, bin_step_in_tuple: Tuple[float, ...]) -> Tuple[int, ...]:
        pass
