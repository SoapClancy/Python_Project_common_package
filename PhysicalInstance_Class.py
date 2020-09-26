import pandas as pd
from Ploting.fast_plot_Func import *
from typing import Tuple, Iterable, Callable, List
from collections import ChainMap
from Filtering.OutlierAnalyser_Class import DataCategoryNameMapper, DataCategoryData
from ConvenientDataType import StrOneDimensionNdarray
import copy


class PhysicalInstance:
    __slots__ = ("obj_name", "predictor_names", "dependant_names")
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `SomeData`
    _metadata = ["obj_name", "predictor_names", "dependant_names"]

    def __init__(self, *args,
                 obj_name: str,
                 predictor_names: Iterable[str],
                 dependant_names: Iterable[str],
                 **kwargs):
        if not (isinstance(self, PhysicalInstanceSeries) or isinstance(self, PhysicalInstanceDataFrame)):
            raise Exception("'PhysicalInstance' should not be called directly, "
                            "please use 'PhysicalInstanceSeries' or 'PhysicalInstanceDataFrame' or any other subclass.")

        super().__init__(*args, **kwargs)
        self.obj_name = obj_name  # type: str
        self.predictor_names = tuple(predictor_names)  # type: tuple
        self.dependant_names = tuple(dependant_names)  # type: tuple

    def init_from_self(self, *args, **kwargs):
        """
        To initialise a new instance from an existing instance, which may a cause of resampling, slicing, etc.
        """
        kwargs = dict(ChainMap(kwargs,
                               {key: self.__getattribute__(key) for key in self.__slots__}))
        new_instance = self.__class__(
            *args,
            obj_name=self.obj_name,
            predictor_names=self.predictor_names,
            dependant_names=self.dependant_names,
            **kwargs)
        return new_instance

    def __str__(self):
        t1 = eval("self.index[0].strftime('%Y-%m-%d %H.%M')")
        t2 = eval("self.index[-1].strftime('%Y-%m-%d %H.%M')")
        return f"{self.obj_name} {self.__class__.__name__} from {t1} to {t2}"

    # %%
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    """
    This section is about data category and outliers 
    """

    @property
    def outlier_name_mapper(self) -> DataCategoryNameMapper:
        """
        Note that this is to analyse all "predictor_names" and "dependant_names" dimensions together,
        i.e., these dimensions are treated as a whole.
        If detailed analysis required for any individual features (i.e., columns), then separate analysis required
        :return:
        """
        meta = [["missing data", "missing", -1, "N/A"],
                ["others data", "others", 0, "N/A"]]
        mapper = DataCategoryNameMapper.init_from_template(rows=len(meta))
        mapper[:] = meta
        return mapper

    def data_category_plot(self: Union[pd.DataFrame, pd.Series], data_category_data: DataCategoryData, ax=None):
        if len(self.predictor_names) + len(self.dependant_names) > 2:
            raise NotImplementedError
        for this_category_abbreviation in np.unique(data_category_data.abbreviation):
            this_category_abbreviation_index = data_category_data.abbreviation == this_category_abbreviation
            if self[this_category_abbreviation_index].shape[0] == 0:
                pass
            else:
                ax = scatter(self[this_category_abbreviation_index][self.predictor_names[0]].values,
                             self[this_category_abbreviation_index][self.dependant_names[0]].values,
                             ax=ax,
                             label=this_category_abbreviation)
        return ax

    def data_category_inside_boundary(self: Union[pd.DataFrame, pd.Series],
                                      boundary: dict, *,
                                      inclusive: bool = True) -> ndarray:
        """
        Define a (super) rectangle boundary, any value inside the boundary will return True

        :param boundary: a dict.
        The key should be str, representing the column concerned.
        The value should be Tuple[Union[int, float], Union[int, float]], representing the min boundary and max boundary.

        :param inclusive: Include boundaries.

        :return: Boolean ndarray.
        """
        pass
        boolean_array = np.full(self.__getattribute__('shape')[0], fill_value=True)
        for this_dim, this_dim_limit in boundary.items():
            boolean_array = np.bitwise_and(boolean_array,
                                           self[this_dim].between(*this_dim_limit, inclusive=inclusive).values)
        return boolean_array

    def data_category_is_linearity(self: Union[pd.DataFrame, pd.Series],
                                   *rolling_args,
                                   constant_error: dict = None,
                                   general_linearity_error: dict = None,
                                   **rolling_kwargs) -> ndarray:
        """
        Try to find linearity (e.g., constant) series within rolling window

        :param constant_error: a dict.
        The key should be str, representing the column concerned.
        The value should be the absolute error that can be tolerated.

        :param general_linearity_error: The same as constant_error. But if specified, it will detect general linearity.

        :return: Boolean ndarray.
        """
        boolean_array = np.full(self.__getattribute__('shape')[0], fill_value=False)
        # %% constant mode
        if constant_error is not None:
            rolling_obj = self.concerned_data().pd_view().rolling(*rolling_args, **rolling_kwargs)
            desired_number = int(np.nanmax(rolling_obj.count().values))
            for this_dim, this_dim_error in constant_error.items():
                this_boolean_array = np.isclose(rolling_obj.min()[this_dim], rolling_obj.max()[this_dim],
                                                rtol=0, atol=this_dim_error)
                this_boolean_array[~np.all(rolling_obj.count() == desired_number, axis=1)] = False
                this_boolean_array_protect = copy.deepcopy(this_boolean_array)
                for i in range(desired_number):
                    this_boolean_array = np.bitwise_or(this_boolean_array, np.roll(this_boolean_array_protect, -i))
                boolean_array = np.bitwise_or(boolean_array, this_boolean_array)
            # boolean_array[~np.all(rolling_obj.count() == desired_number, axis=1)] = False
        # %% general linearity mode
        elif general_linearity_error is not None:
            # TODO 1) desired_number, 2) engine='numba', 3) ~np.all(rolling_obj.count() == desired_number, axis=1)
            def func(x, tol):
                double_diff = np.diff(x, 2)
                if double_diff.shape[0] == 0:
                    return False
                else:
                    return np.max(double_diff) - np.min(double_diff) < tol

            for this_dim, this_dim_error in general_linearity_error.items():
                rolling_obj = self[this_dim].rolling(*rolling_args, **rolling_kwargs)
                this_boolean_array = rolling_obj.apply(func, raw=True,
                                                       # engine='numba',
                                                       args=(this_dim_error,)).astype(bool)
                boolean_array = np.bitwise_or(boolean_array, this_boolean_array)
        else:
            raise Exception("Must specify 'constant_error' or 'general_linearity_error'")

        return boolean_array

    def outlier_detector_initialiser(self, data_category_data_type: str = 'U16') -> DataCategoryData:
        """
        This function is to detect common-found types of outliers
        :param data_category_data_type:
        :return:
        """
        outlier_name_mapper = self.outlier_name_mapper
        outlier = DataCategoryData(abbreviation=StrOneDimensionNdarray(['others'] * self.__getattribute__('shape')[0]),
                                   name_mapper=outlier_name_mapper,
                                   index=self.__getattribute__('index').values)
        # give it enough memory to store the string. "missing" needs > U7. The default now is U10
        outlier.abbreviation = outlier.abbreviation.astype(data_category_data_type)
        # make sure only predictor_names and dependant_names are analysed
        concerned_data = self.concerned_data()
        # %% Missing value
        outlier.abbreviation[concerned_data.isna().any(axis=1).values] = "missing"
        return outlier

    def outlier_detector(self, *args, **kwargs) -> DataCategoryData:
        return self.outlier_detector_initialiser(*args, **kwargs)

    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    def head(self, n=None):
        return pd.DataFrame(self).head(n)

    def tail(self, n=None):
        return pd.DataFrame(self).head(n)

    def describe(self, percentiles=None, include=None, exclude=None):
        return pd.DataFrame(self).describe(percentiles=percentiles,
                                           include=include,
                                           exclude=exclude)

    def pd_view(self):
        if isinstance(self, PhysicalInstanceSeries):
            return pd.Series(self)
        else:
            return pd.DataFrame(self)

    def concerned_data(self: Union[pd.DataFrame, pd.Series]):
        """
        This function return a slice of the instance, where the columns only contain predictor_names and dependant_names
        :return:
        """
        concerned_dims = list(self.predictor_names) + list(self.dependant_names)
        data_obj = self.init_from_self(self[concerned_dims])
        return data_obj

    def resample(self, *resample_args,
                 resampler_obj_func_source_code: str = None,
                 **resample_kwargs):
        """
        :param resampler_obj_func_source_code: str. To be executed right after obtaining the resampler obj.
        :return:
        """
        resampler_obj = self.pd_view().resample(*resample_args, **resample_kwargs)
        if resampler_obj_func_source_code is not None:
            exec("".join(("results = resampler_obj.",
                          resampler_obj_func_source_code)))
            resampler_results = self.init_from_self(locals()['results'])
            return resampler_results
        else:
            return resampler_obj


class PhysicalInstanceSeries(PhysicalInstance, pd.Series):

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return PhysicalInstanceSeries(*args,
                                          obj_name=self.obj_name,
                                          predictor_names=self.predictor_names,
                                          dependant_names=self.dependant_names,
                                          **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_expanddim(self):
        def _c(*args, **kwargs):
            return PhysicalInstanceDataFrame(*args,
                                             obj_name=self.obj_name,
                                             predictor_names=self.predictor_names,
                                             dependant_names=self.dependant_names,
                                             **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_sliced(self):
        raise NotImplementedError(f"Slice is not supported for {self.__class__.__name__}!")


class PhysicalInstanceDataFrame(PhysicalInstance, pd.DataFrame):
    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return self.init_from_self(*args, **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError(f"Expanddim is not supported for {self.__class__.__name__}!")

    @property
    def _constructor_sliced(self):
        def _c(*args, **kwargs):
            return PhysicalInstanceSeries(*args,
                                          obj_name=self.obj_name,
                                          predictor_names=self.predictor_names,
                                          dependant_names=self.dependant_names,
                                          **kwargs).__finalize__(self)

        return _c

    def unique(self, subset=None):
        subset = self[subset] if subset is not None else self
        duplicated_mask = subset.duplicated()
        unique_rows = subset[~duplicated_mask.values]

        unique_label = np.full(subset.shape[0], 0)
        for i, (index, this_row) in enumerate(unique_rows.iterrows()):
            eq_mask = np.all(this_row.values == subset.values, axis=1)
            unique_label[eq_mask] = i
        return self[~duplicated_mask], unique_label


if __name__ == '__main__':
    tt_df = PhysicalInstanceDataFrame(np.arange(120).reshape((8, 15)),
                                      obj_name='tt_name',
                                      predictor_names=('tt_predictor_names',),
                                      dependant_names=('tt_dependant_names',))
    tt_series = tt_df[0]
