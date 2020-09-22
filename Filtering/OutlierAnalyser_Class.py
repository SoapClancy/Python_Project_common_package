from Ploting.fast_plot_Func import *
import pandas as pd
from typing import Tuple, Iterable
from ConvenientDataType import StrOneDimensionNdarray
from pathlib import Path
import copy


class DataCategoryNameMapperMeta(type):
    @staticmethod
    def _make_infer_from_funcs(func_names: Tuple[str], infer_names: Tuple[str]):
        codes = []
        for i in range(func_names.__len__()):
            codes.append(
                f"def infer_from_{func_names[i]}(self, value: Union[str, int]) -> pd.DataFrame:\n"
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

    def convert_sequence_data_key(self, old_key_name: str, new_key_name: str, *, sequence_data):
        """
        An vectorised version of infer_from_...
        :param old_key_name: str, can be one in ['long name', 'abbreviation', 'code', 'description']
        :param new_key_name: str, can be one in ['long name', 'abbreviation', 'code', 'description']
        :param sequence_data
        :return:
        """
        assert all((old_key_name in DataCategoryNameMapper.columns_template,
                    new_key_name in DataCategoryNameMapper.columns_template,
                    old_key_name != new_key_name))
        if old_key_name == 'long name':
            func = self.__getattr__("infer_from_long_name")
        else:
            func = self.__getattr__(f"self.infer_from_{old_key_name}")

        sequence_data = copy.deepcopy(sequence_data)
        for this_unique_sequence_data in np.unique(sequence_data):
            this_unique_sequence_data_mask = sequence_data == this_unique_sequence_data
            sequence_data[this_unique_sequence_data_mask] = func(this_unique_sequence_data)[new_key_name].values[0]
        return sequence_data


class DataCategoryData:
    __slots__ = ('abbreviation', 'index', 'name_mapper')

    def __init__(self, abbreviation: StrOneDimensionNdarray = None, *,
                 index: Union[ndarray, pd.Index] = None,
                 name_mapper: DataCategoryNameMapper):
        """
        :param abbreviation: abbreviation of category.
        Strongly recommended to use StrOneDimensionNdarray for all purposes including indexing and setting values

        :param index: The index of data, optional, which is usually the same as the analysed obj

        :param name_mapper:
        """
        self.abbreviation = StrOneDimensionNdarray(abbreviation) if abbreviation is not None else None
        self.index = index if index is not None else StrOneDimensionNdarray(['N/A index'])
        self.name_mapper = name_mapper  # type: DataCategoryNameMapper

    def rename(self, mapper: dict, new_name_mapper: DataCategoryNameMapper):
        self.name_mapper = new_name_mapper
        for key, val in mapper.items():
            self.abbreviation[self(key)] = val
        return self

    @property
    def pd_view(self):
        return pd.DataFrame(self.abbreviation, index=self.index, columns=['category abbreviation'])

    def __call__(self, key: Union[Iterable[str], str]):
        return np.isin(self.abbreviation, key)

    def report(self, report_pd_to_csv_file_path: Path = None, *,
               abbreviation_rename_mapper: dict = None,
               sorted_kwargs: dict = None):
        abbreviation_unique = np.unique(self.abbreviation)
        abbreviation_unique = sorted(abbreviation_unique, **(sorted_kwargs or {}))
        report_pd = pd.DataFrame(index=abbreviation_unique,
                                 columns=['number', 'percentage'], dtype=float)
        for this_outlier in abbreviation_unique:
            this_outlier_number = sum(self(this_outlier))
            report_pd.loc[this_outlier, 'number'] = this_outlier_number
            report_pd.loc[this_outlier, 'percentage'] = this_outlier_number / self.abbreviation.shape[0] * 100
        report_pd.rename(index=abbreviation_rename_mapper or {}, inplace=True)
        bar(report_pd.index, report_pd['percentage'].values, y_label="Recording percentage [%]",
            autolabel_format="{:.2f}", y_lim=(-1, 85))
        plt.xticks(rotation=45)

        bar(report_pd.index, report_pd['number'].values, y_label="Recording number",
            autolabel_format="{:.0f}", y_lim=(-1, np.max(report_pd['number'].values) * 1.2))
        plt.xticks(rotation=45)
        if report_pd_to_csv_file_path is not None:
            report_pd.to_csv(report_pd_to_csv_file_path)
            self.name_mapper.to_csv(report_pd_to_csv_file_path.parent / 'name_mapper.csv')
            pd.merge(report_pd, self.name_mapper.pd_view, how='left',
                     left_index=True,
                     right_on='abbreviation').to_csv(report_pd_to_csv_file_path.parent / 'report_with_name_mapper.csv')


if __name__ == '__main__':
    tt = DataCategoryNameMapper.init_from_template(rows=2)
    tt.iloc[0] = ['CAT-I', 'CAT-I', 1, 'CAT-I de']
    tt.iloc[1] = ['CAT-II', 'CAT-II', 2, 'CAT-II de']
    tt.infer_from_code(1)
