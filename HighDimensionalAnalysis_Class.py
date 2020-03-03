from numpy import ndarray
from sklearn.cluster import KMeans
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from Ploting.fast_plot_Func import scatter
from typing import Tuple


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
