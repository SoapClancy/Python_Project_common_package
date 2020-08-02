import pandas as pd
import numpy as np
from numpy import ndarray
from Ploting.fast_plot_Func import *
from typing import Tuple


class PhysicalInstance(pd.DataFrame):
    __slots__ = ("data", "obj_name", "predictor_names", "dependant_names")
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `SomeData`
    _metadata = ["obj_name", "predictor_names", "dependant_names"]

    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finlaize__` every time."""

        def _c(*args, **kwargs):
            return PhysicalInstance(*args,
                                    obj_name=self.obj_name,
                                    predictor_names=self.predictor_names,
                                    dependant_names=self.dependant_names,
                                    **kwargs).__finalize__(self)

        return _c

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError("Not supported for PhysicalInstance!")

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, *args,
                 obj_name: str,
                 predictor_names: Tuple[str, ...],
                 dependant_names: Tuple[str, ...],
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.obj_name = obj_name  # type: str
        import warnings
        warnings.simplefilter(action='ignore')
        setattr(self, 'predictor_names', predictor_names)
        setattr(self, 'dependant_names', dependant_names)
        warnings.simplefilter(action='default')

    def __getitem__(self, item):
        if isinstance(item, int):
            pass
        elif isinstance(item, slice):
            pass
        else:
            raise NotImplementedError("PhysicalInstance instance __getitem__() should be type 'int',"
                                      "or type 'slice'")


if __name__ == '__main__':
    tt = PhysicalInstance(np.arange(120).reshape((8, 15)),
                          obj_name='tt_name',
                          predictor_names=('tt_predictor_names',),
                          dependant_names=('tt_dependant_names',))
    aa = tt[0]
