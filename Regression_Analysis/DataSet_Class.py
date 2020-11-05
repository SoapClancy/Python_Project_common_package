from typing import Tuple
import pandas as pd
from Ploting.fast_plot_Func import *
from pathlib import Path
import tensorflow as tf
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear
from itertools import chain


class DeepLearningDataSet:
    __slots__ = ('data', 'name', 'transformed_cols_meta', 'transformed_data', 'predictor_cols', 'dependant_cols')

    def __init__(self, original_data_set: pd.DataFrame, *,
                 name: str,
                 cos_sin_transformed_col: Tuple[str, ...] = (),
                 min_max_transformed_col: Tuple[str, ...] = (),
                 one_hot_transformed_col: Tuple[str, ...] = (),
                 non_transformed_col: Tuple[str, ...] = (),
                 predictor_cols: Tuple[str, ...] = (),
                 dependant_cols: Tuple[str, ...] = (),
                 transformation_args_folder_path: Path,
                 stacked_shift_col: Tuple[str, ...] = (),
                 stacked_shift_size: Tuple[datetime.timedelta, ...] = (),
                 how_many_stacked: Tuple[int, ...] = ()):
        assert ('training' in name) or ('test' in name)
        assert (predictor_cols or dependant_cols) is not ()
        assert not ((predictor_cols is not ()) and (dependant_cols is not ()))

        self.data = original_data_set  # type: pd.DataFrame
        self.name = name
        self.transformed_cols_meta = {
            'cos_sin_transformed_col': cos_sin_transformed_col,
            'min_max_transformed_col': min_max_transformed_col,
            'one_hot_transformed_col': one_hot_transformed_col,
            'non_transformed_col': non_transformed_col
        }
        self.predictor_cols, self.dependant_cols = self._infer_predictor_and_dependant_cols(predictor_cols,
                                                                                            dependant_cols)
        self.transformed_data = self._preprocess(transformation_args_folder_path)
        # The further step is to stack the shifts
        self.transformed_data = self._stacked_shift(stacked_shift_col, stacked_shift_size, how_many_stacked)

    @property
    def considered_cols_list(self):
        return list(chain(*self.transformed_cols_meta.values()))

    def _infer_predictor_and_dependant_cols(self, predictor_cols, dependant_cols):
        if predictor_cols is not ():
            return predictor_cols, list(set(self.considered_cols_list) - set(predictor_cols))
        else:
            return list(set(self.considered_cols_list) - set(dependant_cols)), dependant_cols

    def _preprocess(self, transformation_args_folder_path: Path = None):
        """
        This function
        """
        name = self.name.replace("test", "training")
        transformation_args_file_path = transformation_args_folder_path / (name + '.pkl')

        # %% Obtain the args for transformation
        @load_exist_pkl_file_otherwise_run_and_save(transformation_args_file_path)
        def load_transformation_args_func():
            _transformed_cols_args = {key: None for key in self.transformed_cols_meta}
            _new_multi_index_columns = {key: None for key in self.considered_cols_list}
            # %% min_max_transformed_col
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(self.data[list(self.transformed_cols_meta['min_max_transformed_col'])].values)
            _transformed_cols_args['min_max_transformed_col'] = min_max_scaler
            for this_col in self.transformed_cols_meta['min_max_transformed_col']:
                _new_multi_index_columns[this_col] = [(this_col, 'min_max')]

            # %% cos_sin_transformed_col
            cos_sin_transformed_col_args = {key: None for key in self.transformed_cols_meta['cos_sin_transformed_col']}
            for this_col in self.transformed_cols_meta['cos_sin_transformed_col']:
                this_col_data = self.data[this_col].values
                this_col_lower_boundary = np.nanmin(this_col_data)
                this_col_upper_boundary = np.nanmax(this_col_data)
                cos_sin_transformed_col_args[this_col] = (this_col_lower_boundary, this_col_upper_boundary)
                _new_multi_index_columns[this_col] = [(this_col, 'cos'), (this_col, 'sin')]
            _transformed_cols_args['cos_sin_transformed_col'] = cos_sin_transformed_col_args

            # %% one_hot_transformed_col
            if self.transformed_cols_meta['one_hot_transformed_col'] is not ():
                one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
                one_hot_encoder.fit(self.data[list(self.transformed_cols_meta['one_hot_transformed_col'])].values)
                for i, this_col in enumerate(self.transformed_cols_meta['one_hot_transformed_col']):
                    _new_multi_index_columns[this_col] = [(this_col, f'one_hot_{x}') for x in
                                                          one_hot_encoder.categories_[i]]
                _transformed_cols_args['one_hot_transformed_col'] = one_hot_encoder

            # %% non_transformed_col
            for this_col in self.transformed_cols_meta['non_transformed_col']:
                _new_multi_index_columns[this_col] = [(this_col, 'original')]

            return _transformed_cols_args, _new_multi_index_columns

        transformed_cols_args, new_multi_index_columns = load_transformation_args_func()
        # %% Do transformation
        transformed_data = pd.DataFrame(
            index=self.data.index,
            columns=pd.MultiIndex.from_tuples(reduce(lambda a, b: a + b, new_multi_index_columns.values()),
                                              names=('feature', 'notes')),
            dtype=float
        )
        # %% min_max_transformed_col and
        _ = 'min_max_transformed_col'
        _index = list(self.transformed_cols_meta[_])
        trans = transformed_cols_args[_].transform(self.data[_index])
        transformed_data.loc[:, _index] = trans
        # %% one_hot_transformed_col
        if self.transformed_cols_meta['one_hot_transformed_col'] is not ():
            _ = 'one_hot_transformed_col'
            _index = list(self.transformed_cols_meta[_])
            trans = transformed_cols_args[_].transform(self.data[_index])
            transformed_data.loc[:, _index] = trans.toarray().astype(int)

        # %% cos_sin_transformed_col
        _index = list(self.transformed_cols_meta['cos_sin_transformed_col'])
        for this_index in _index:
            obj = CircularToLinear(lower_boundary=transformed_cols_args['cos_sin_transformed_col'][this_index][0],
                                   upper_boundary=transformed_cols_args['cos_sin_transformed_col'][this_index][1],
                                   period=transformed_cols_args['cos_sin_transformed_col'][this_index][1])
            trans = obj.transform(self.data[this_index].values)
            transformed_data.loc[:, (this_index, 'cos')] = trans['cos']
            transformed_data.loc[:, (this_index, 'sin')] = trans['sin']

        # %% non_transformed_col
        _index = list(self.transformed_cols_meta['non_transformed_col'])
        transformed_data.loc[:, _index] = self.data[list(self.transformed_cols_meta['non_transformed_col'])].values
        return transformed_data

    def _stacked_shift(self, stacked_shift_col, stacked_shift_size, how_many_stacked):
        if (stacked_shift_col or stacked_shift_size or how_many_stacked) is ():
            return self.transformed_data
        else:
            for (this_stacked_shift_col, this_stacked_shift_size, this_how_many_stacked) in zip(
                    stacked_shift_col, stacked_shift_size, how_many_stacked):
                assert self.transformed_data[this_stacked_shift_col].shape[1] == 1
                to_shift = self.transformed_data[this_stacked_shift_col]
                for i in range(1, this_how_many_stacked + 1):
                    this_multi_index = pd.MultiIndex.from_tuples([(this_stacked_shift_col, f"shift_{i}")],
                                                                 names=self.transformed_data.columns.names)
                    shift = to_shift.shift(periods=i, freq=this_stacked_shift_size)
                    shift.columns = this_multi_index
                    self.transformed_data = pd.merge(self.transformed_data, shift, how='left',
                                                     left_index=True, right_index=True)
            return self.transformed_data

    def windowed_dataset(self, window_length: datetime.timedelta, *,
                         drop_remainder=False,
                         batch_size: int,
                         shift=None) -> tf.data.Dataset:
        assert np.unique(np.diff(self.transformed_data.index.values)).size == 1
        freq = (self.transformed_data.index.values[1] - self.transformed_data.index.values[0]) / np.timedelta64(1, 's')
        window_size = int(window_length.total_seconds() / freq)

        transformed_data_ndarray = self.transformed_data.values
        predictor_cols_index = np.unique(list(
            chain(*[list(self.transformed_data.columns.__getattribute__('get_locs')([x]))
                    for x in self.predictor_cols],
                  *[[i] for i, col in enumerate(self.transformed_data.columns) if 'shift' in col[1]])
        ))
        dependant_cols_index = list(chain(*[list(self.transformed_data.columns.__getattribute__('get_locs')([x]))
                                            for x in self.dependant_cols]))
        dependant_cols_index = [i for i in dependant_cols_index if 'shift' not in self.transformed_data.columns[i][1]]
        assert (predictor_cols_index.__len__() + dependant_cols_index.__len__()) == self.transformed_data.shape[1]

        predictor_cols_index.sort()
        dependant_cols_index.sort()

        # %% Remove any nan
        i = 0
        remove_mask = np.full(len(transformed_data_ndarray), False)
        while True:
            i_next = i + window_size
            if np.any(np.isnan(transformed_data_ndarray[i:min((i_next, transformed_data_ndarray.shape[0]))])):
                remove_mask[i:min((i_next, transformed_data_ndarray.shape[0]))] = True
            if i_next > transformed_data_ndarray.shape[0]:
                break
            i = i_next

        transformed_data_ndarray = transformed_data_ndarray[~remove_mask]

        transformed_data_ndarray = tf.data.Dataset.from_tensor_slices(transformed_data_ndarray)
        transformed_data_ndarray = transformed_data_ndarray.window(window_size,
                                                                   shift=shift or window_size,
                                                                   drop_remainder=drop_remainder)
        transformed_data_ndarray = transformed_data_ndarray.flat_map(
            lambda window: window.batch(window_size, drop_remainder=drop_remainder)
        )
        transformed_data_ndarray = transformed_data_ndarray.map(
            lambda window: (tf.gather(window, predictor_cols_index, axis=1),
                            tf.gather(window, dependant_cols_index, axis=1))
        )
        transformed_data_ndarray = transformed_data_ndarray.batch(batch_size).prefetch(2)
        return transformed_data_ndarray
