import pandas as pd
from Ploting.fast_plot_Func import *
from typing import Tuple, Iterable


class PhysicalInstance:
    __slots__ = ("data", "obj_name", "predictor_names", "dependant_names")
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
        self.predictor_names = predictor_names  # type: Iterable
        self.dependant_names = dependant_names  # type: Iterable

    def head(self, n=None):
        return pd.DataFrame(self).head(n)

    def tail(self, n=None):
        return pd.DataFrame(self).head(n)

    def describe(self, percentiles=None, include=None, exclude=None):
        return pd.DataFrame(self).describe(
            percentiles=percentiles,
            include=include,
            exclude=exclude,
        )

    def pd_view(self):
        if isinstance(self, PhysicalInstanceSeries):
            return pd.Series(self)
        else:
            return pd.DataFrame(self)

    def __str__(self):
        t1 = eval("self.index[0].strftime('%Y-%m-%d %H.%M')")
        t2 = eval("self.index[-1].strftime('%Y-%m-%d %H.%M')")
        return f"{self.obj_name} {self.__class__.__name__} from {t1} to {t2}"


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
            return self.__class__(*args,
                                  obj_name=self.obj_name,
                                  predictor_names=self.predictor_names,
                                  dependant_names=self.dependant_names,
                                  **kwargs).__finalize__(self)

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


if __name__ == '__main__':
    tt_df = PhysicalInstanceDataFrame(np.arange(120).reshape((8, 15)),
                                      obj_name='tt_name',
                                      predictor_names=('tt_predictor_names',),
                                      dependant_names=('tt_dependant_names',))
    tt_series = tt_df[0]