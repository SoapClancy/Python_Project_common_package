from Ploting.fast_plot_Func import *
import pandas as pd
from typing import Tuple, Iterable
from ConvenientDataType import IntOneDimensionNdarray, StrOneDimensionNdarray


class DataCategoryNameMapperMeta(type):
    @staticmethod
    def _make_infer_from_funcs(func_names: Tuple[str], infer_names: Tuple[str]):
        codes = []
        for i in range(func_names.__len__()):
            codes.append(
                f"def infer_from_{func_names[i]}(self, value: Union[str, int]) -> pd.DataFrame:\n" + \
                f"    return self[self['{infer_names[i]}'] == value]"
            )
        return codes

    def __new__(mcs, name, bases, clsdict):
        infer_from_funcs_source_codes = mcs._make_infer_from_funcs(
            tuple(['long_name'] + clsdict['columns_template'][1:]),
            tuple(clsdict['columns_template'])
        )
        for this_infer_from_funcs_source_code in infer_from_funcs_source_codes:
            exec(this_infer_from_funcs_source_code, globals(), clsdict)
        clsobj = super().__new__(mcs, name, bases, clsdict)
        return clsobj


class DataCategoryNameMapper(pd.DataFrame, metaclass=DataCategoryNameMapperMeta):
    columns_template = ['long name', 'abbreviation', 'code', 'description']

    @property
    def _constructor(self):
        return self.__class__

    @property
    def _constructor_expanddim(self):
        return NotImplementedError(f"Expanddim is not supported for {self.__class__.__name__}!")

    @property
    def _constructor_sliced(self):
        return pd.Series

    @property
    def pd_view(self):
        return pd.DataFrame(self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: pd.DataFrame
        assert (list(self.columns) == self.columns_template), f"Columns of {self.__class__.__name__} instance " \
                                                              f"must be {self.columns_template}"
        assert (all(
            (pd.api.types.is_string_dtype(self.iloc[:, 0]),
             pd.api.types.is_string_dtype(self.iloc[:, 1]),
             pd.api.types.is_string_dtype(self.iloc[:, 3]))
        )), f"{self.columns_template[:-2] + self.columns_template[-1:]} must be str"
        assert (pd.api.types.is_integer_dtype(self.iloc[:, 2])), f"{self.columns_template[-2]} must be int"

    @classmethod
    def init_from_template(cls,
                           long_name: str = '',
                           abbreviation: str = '',
                           code: int = 0,
                           description: str = '', *,
                           rows: int = 1):
        template_obj = pd.DataFrame({
            cls.columns_template[0]: long_name,
            cls.columns_template[1]: abbreviation,
            cls.columns_template[2]: code,
            cls.columns_template[3]: description
        }, index=range(rows))
        return cls(template_obj)


class DataCategoryData:
    __slots__ = ('abbreviation', 'index', 'name_mapper')

    def __init__(self, data: StrOneDimensionNdarray = None, *,
                 index: ndarray = None,
                 name_mapper: DataCategoryNameMapper = None):
        """
        Strongly recommended to use StrOneDimensionNdarray for all purposes including indexing and setting values
        :param data: abbreviation of category
        :param name_mapper:
        """
        self.abbreviation = StrOneDimensionNdarray(data) if data is not None else None
        self.index = index if index is not None else StrOneDimensionNdarray(['N/A index'])
        self.name_mapper = name_mapper  # type: DataCategoryNameMapper

    @property
    def pd_view(self):
        return pd.DataFrame(self.abbreviation, index=self.index, columns=['category abbreviation'])

    def __call__(self, abbreviation: Union[Iterable[str], str]):
        return np.isin(self.abbreviation, abbreviation)


if __name__ == '__main__':
    tt = DataCategoryNameMapper.init_from_template(rows=2)
    tt.iloc[0] = ['CAT-I', 'CAT-I', 1, 'CAT-I de']
    tt.iloc[1] = ['CAT-II', 'CAT-II', 2, 'CAT-II de']
    tt.infer_from_code(1)
