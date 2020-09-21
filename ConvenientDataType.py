import numpy as np
from typing import Iterable, Union, Tuple
from numpy import ndarray
import pandas as pd
from scipy.stats import norm
import warnings
from Ploting.fast_plot_Func import *
from itertools import chain
import re


class ComplexNdarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.dtype != 'complex':
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class OneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.ndim != 1:
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class IntOneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        try:
            obj = OneDimensionNdarray(input_array)
            if obj.dtype != int:
                raise TypeError
        except TypeError:
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class StrOneDimensionNdarray(np.ndarray):

    def __new__(cls, input_array):
        try:
            obj = OneDimensionNdarray(input_array)
            if not isinstance(obj[0], str):
                if (isinstance(obj[0], float)) or (isinstance(obj[0], int)):
                    obj = obj.astype(str)
                else:
                    raise TypeError
        except TypeError:
            raise TypeError(f"Expected {cls}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class IntFloatConstructedOneDimensionNdarray(np.ndarray):

    def __new__(cls, int_or_float_or_iterable):
        if isinstance(int_or_float_or_iterable, int) or isinstance(int_or_float_or_iterable, float):
            obj = np.asarray([int_or_float_or_iterable]).view(cls)
        else:
            obj = np.asarray(int_or_float_or_iterable).flatten().view(cls)
        obj = obj.astype(float)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        pass


class UncertaintyDataFrame(pd.DataFrame):
    __slots__ = ()

    @property
    def _constructor(self):
        return UncertaintyDataFrame

    @property
    def _constructor_expanddim(self):
        return NotImplementedError(f"Expanddim is not supported for {self.__class__.__name__}!")

    @property
    def _constructor_sliced(self):
        return pd.Series

    @property
    def pd_view(self):
        return pd.DataFrame(self)

    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, dtype='float', **kwargs)
        # The index must be str
        StrOneDimensionNdarray(self.index.values)
        # The column can be either str or int
        if not pd.api.types.is_integer_dtype(self.columns):
            StrOneDimensionNdarray(self.columns.values)
        if (self.index.values[-2] != 'mean') or (self.index.values[-1] != 'std.'):
            raise Exception("UncertaintyDataFrame must use StrOneDimensionNdarray as index, "
                            "and the last two indices should be 'mean' and 'std.'")

    @classmethod
    def init_from_template(cls, columns_number: int, *,
                           percentiles: Union[None, ndarray] = np.arange(0, 100.001, 0.001)):
        from Data_Preprocessing.float_precision_control_Func import covert_to_str_one_dimensional_ndarray
        if percentiles is None:
            percentiles = []
        else:
            percentiles = covert_to_str_one_dimensional_ndarray(percentiles, '0.001')
        uncertainty_dataframe = pd.DataFrame(
            index=list(chain(percentiles,
                             np.array([[f'{x}_Sigma_low', f'{x}_Sigma_high'] for x in (1, 1.5, 2, 3, 4.5)]).flatten(),
                             ['mean', 'std.'])),
            columns=range(columns_number)
        )
        return cls(uncertainty_dataframe)

    @classmethod
    def init_from_2d_ndarray(cls, two_dim_array: ndarray, *, percentiles: ndarray = np.arange(0, 100.001, 0.001)):
        uncertainty_dataframe = cls.init_from_template(two_dim_array.shape[1],
                                                       percentiles=percentiles)
        for i in range(two_dim_array.shape[1]):
            uncertainty_dataframe.iloc[:, i] = np.concatenate(
                (np.percentile(two_dim_array[:, i], percentiles.astype('float')),
                 np.mean(two_dim_array[:, i], keepdims=True),
                 np.std(two_dim_array[:, i], keepdims=True))
            )
        return cls(uncertainty_dataframe)

    def infer_higher_half_percentiles(self, lower_half_percentiles: StrOneDimensionNdarray) -> StrOneDimensionNdarray:
        """
        Use lower half percentiles to infer higher half percentiles
        """
        higher_half_percentiles = []
        for this_lower_half_percentile in lower_half_percentiles:
            this_higher_half_percentile_index = np.argmin(
                np.abs(self.index.values[:-2].astype(np.float) - (100 - float(this_lower_half_percentile)))
            )
            higher_half_percentiles.append(self.index.values[this_higher_half_percentile_index])
        return StrOneDimensionNdarray(higher_half_percentiles)

    @staticmethod
    def infer_percentile_boundaries_by_sigma(by_sigma: Union[int, float, str]) -> Tuple[float, float]:
        by_sigma = float(by_sigma)
        preserved_data_percentage = 100 * (norm.cdf(by_sigma) - norm.cdf(-1 * by_sigma))
        return (100 - preserved_data_percentage) / 2, 100 - (100 - preserved_data_percentage) / 2

    def __call__(self, preserved_data_percentage: Union[int, float] = None, *,
                 by_sigma: Union[int, float] = None):
        """
        This __call__ is to select the required percentage of data. For example, if x% data are to be preserved,
        then any data outside [(1-x)/2, 1-(1-x)/2] are to be preserved.
        Since self is a UncertaintyDataFrame obj, it will gives two rows of data, whose indices are the closest to
        (1-x)/2 and 1-x/2

        :param preserved_data_percentage:
        :param by_sigma:
        :return:
        """
        error_msg = "Should specify either 'preserved_data_percentage' or 'preserved_data_percentage_by_sigma'"
        assert (preserved_data_percentage != by_sigma), error_msg
        if by_sigma is not None:
            preserved_data_percentage = self.infer_percentile_boundaries_by_sigma(by_sigma)
        else:
            preserved_data_percentage = [(100 - preserved_data_percentage) / 2,
                                         100 - (100 - preserved_data_percentage) / 2]
        index_float = np.array(self.index[:-2]).astype('float')
        # TODO 向量化和插值
        index_select = [np.argmin(np.abs(index_float - x)) for x in preserved_data_percentage]
        return pd.DataFrame(self).iloc[index_select, :]

    def update_one_column(self, column_name: Union[int, str], *, data: Union[OneDimensionNdarray, ndarray]):
        data = OneDimensionNdarray(data)
        percentile_float = []
        for this_index in self.index[:-2]:
            try:
                this_percentile = float(this_index)
            except ValueError:
                this_sigma = re.findall(r".*(?=_Sigma)", this_index)[0]
                if 'low' in this_index:
                    this_percentile = self.infer_percentile_boundaries_by_sigma(this_sigma)[0]
                else:
                    this_percentile = self.infer_percentile_boundaries_by_sigma(this_sigma)[1]
            percentile_float.append(this_percentile)
        self[column_name] = np.concatenate(
            (np.percentile(data, percentile_float),
             np.mean(data, keepdims=True),
             np.std(data, keepdims=True))
        )
        return self[column_name]

    def sigma_percentage_plot(self, ax=None, **kwargs):
        x_plot = np.array(self.columns).astype(float)
        ax = series(x_plot, self.loc['mean'], ax=ax, color=(0, 1, 0), linestyle='--', label='Mean')
        ax = series(x_plot, self(0).iloc[0].values, ax=ax, linestyle='-',
                    color='orange', label='Median')

        ax = series(x_plot, self(by_sigma=1).iloc[0].values, ax=ax, linestyle='-',
                    label='1' + r'$\sigma$' + ' %', color='grey')
        ax = series(x_plot, self(by_sigma=1).iloc[1].values, ax=ax, linestyle='-',
                    color='grey')
        ax = scatter(x_plot, (self(by_sigma=1).iloc[0].values + self(by_sigma=1).iloc[1].values) / 2, ax=ax,
                     label='1' + r'$\sigma$' + ' %' + '\nmean', marker='+', s=32, color='grey')

        ax = series(x_plot, self(by_sigma=2).iloc[0].values, ax=ax, linestyle='--',
                    label='2' + r'$\sigma$' + ' %', color='fuchsia')
        ax = series(x_plot, self(by_sigma=2).iloc[1].values, ax=ax, linestyle='--',
                    color='fuchsia')
        ax = scatter(x_plot, (self(by_sigma=2).iloc[0].values + self(by_sigma=2).iloc[1].values) / 2, ax=ax,
                     label='2' + r'$\sigma$' + ' %' + '\nmean', marker='x', s=16, color='fuchsia')

        ax = series(x_plot, self(by_sigma=3).iloc[0].values, ax=ax, linestyle='-.',
                    label='3' + r'$\sigma$' + ' %', color='royalblue')
        ax = series(x_plot, self(by_sigma=3).iloc[1].values, ax=ax, linestyle='-.',
                    color='royalblue')
        ax = scatter(x_plot, (self(by_sigma=3).iloc[0].values + self(by_sigma=3).iloc[1].values) / 2, ax=ax,
                     label='3' + r'$\sigma$' + ' %' + '\nmean', marker='s', s=10, color='royalblue')

        ax = series(x_plot, self(by_sigma=4.5).iloc[0].values, ax=ax, linestyle=':',
                    label='4.5' + r'$\sigma$' + ' %', color='black')
        ax = series(x_plot, self(by_sigma=4.5).iloc[1].values, ax=ax, linestyle=':',
                    color='black',
                    **kwargs)
        ax = scatter(x_plot, (self(by_sigma=4.5).iloc[0].values + self(by_sigma=4.5).iloc[1].values) / 2, ax=ax,
                     label='4.5' + r'$\sigma$' + ' %' + '\nmean', marker='|', s=10, color='black', **kwargs)
        # plt.gca().legend(ncol=4, loc='upper center', prop={'size': 10})

        return ax
