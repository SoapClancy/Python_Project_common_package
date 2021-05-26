# from __future__ import annotations

from typing import Tuple, Sequence, Generator, Callable, List
import pandas as pd
from Ploting.fast_plot_Func import *
from pathlib import Path
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, QuantileTransformer
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear
from itertools import chain
import warnings
import copy

try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


class DeepLearningDataSet:
    __slots__ = ('data', 'name', 'transformed_cols_meta', 'transformed_data', 'predictor_cols', 'dependant_cols',
                 '_transformation_args_file_path')

    def __init__(self, original_data_set: pd.DataFrame, *,
                 name: str,
                 cos_sin_transformed_col: Tuple[str, ...] = (),
                 min_max_transformed_col: Tuple[str, ...] = (),
                 standard_transformed_col: Tuple[str, ...] = (),
                 quantile_transformed_col: Tuple[str, ...] = (),
                 one_hot_transformed_col: Tuple[str, ...] = (),
                 non_transformed_col: Tuple[str, ...] = (),
                 predictor_cols: Tuple[str, ...] = (),
                 dependant_cols: Tuple[str, ...] = (),
                 transformation_args_folder_path: Path,
                 stacked_shift_col: Tuple[str, ...] = (),
                 stacked_shift_size: Tuple[datetime.timedelta, ...] = (),
                 how_many_stacked: Tuple[int, ...] = ()):
        assert ('training' in name) or ('test' in name)

        self.data = original_data_set  # type: pd.DataFrame
        self.name = name
        self.transformed_cols_meta = {
            'cos_sin_transformed_col': cos_sin_transformed_col,
            'min_max_transformed_col': min_max_transformed_col,
            'standard_transformed_col': standard_transformed_col,
            'quantile_transformed_col': quantile_transformed_col,
            'one_hot_transformed_col': one_hot_transformed_col,
            'non_transformed_col': non_transformed_col
        }
        self.predictor_cols, self.dependant_cols = self._infer_predictor_and_dependant_cols(predictor_cols,
                                                                                            dependant_cols)

        if 'test' in name:
            assert try_to_find_file(transformation_args_folder_path /
                                    (self.name.replace("test", "training") + '_transformation_args.pkl'))

        self._transformation_args_file_path = (transformation_args_folder_path /
                                               (self.name.replace("test", "training") + '_transformation_args.pkl'))

        self.transformed_data = self._preprocess()
        # The further step is to stack the shifts
        self.transformed_data = self._stacked_shift(stacked_shift_col, stacked_shift_size, how_many_stacked)

    def __str__(self):
        return f"{self.name}, from {self.data.index[0]} to {self.data.index[-1]}"

    @property
    def considered_cols_list(self):
        cols = list(chain(*self.transformed_cols_meta.values()))
        assert set(cols) == set(self.data.columns)
        return cols

    def _infer_predictor_and_dependant_cols(self, predictor_cols, dependant_cols):
        if predictor_cols != () and dependant_cols == ():
            return list((predictor_cols, list(set(self.considered_cols_list) - set(predictor_cols))))
        elif predictor_cols == () and dependant_cols != ():
            return list(set(self.considered_cols_list) - set(dependant_cols)), dependant_cols
        elif predictor_cols != () and dependant_cols != ():
            return list((predictor_cols, dependant_cols))
        else:
            raise

    def _preprocess(self):
        # %% Obtain the args for transformation
        @load_exist_pkl_file_otherwise_run_and_save(self._transformation_args_file_path)
        def load_transformation_args_func():
            _transformed_cols_args = {key: None for key in self.transformed_cols_meta}
            _new_multi_index_columns = {key: None for key in self.considered_cols_list}
            # %% min_max_transformed_col
            if self.transformed_cols_meta['min_max_transformed_col'] != ():
                min_max_scaler = MinMaxScaler()
                min_max_scaler.fit(self.data[list(self.transformed_cols_meta['min_max_transformed_col'])].values)
                _transformed_cols_args['min_max_transformed_col'] = min_max_scaler
                for this_col in self.transformed_cols_meta['min_max_transformed_col']:
                    _new_multi_index_columns[this_col] = [(this_col, 'min_max')]

            # %% standard_transformed_col
            if self.transformed_cols_meta['standard_transformed_col'] != ():
                standard_scaler = StandardScaler()
                standard_scaler.fit(self.data[list(self.transformed_cols_meta['standard_transformed_col'])].values)
                _transformed_cols_args['standard_transformed_col'] = standard_scaler
                for this_col in self.transformed_cols_meta['standard_transformed_col']:
                    _new_multi_index_columns[this_col] = [(this_col, 'standard')]

            # %% quantile_transformed_col
            if self.transformed_cols_meta['quantile_transformed_col'] != ():
                quantile_scaler = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
                quantile_scaler.fit(self.data[list(self.transformed_cols_meta['quantile_transformed_col'])].values)
                _transformed_cols_args['quantile_transformed_col'] = quantile_scaler
                for this_col in self.transformed_cols_meta['quantile_transformed_col']:
                    _new_multi_index_columns[this_col] = [(this_col, 'quantile')]

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
            if self.transformed_cols_meta['one_hot_transformed_col'] != ():
                one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
                to_fit = self.data[list(self.transformed_cols_meta['one_hot_transformed_col'])].values
                to_fit = to_fit[np.all(~np.isnan(to_fit), axis=1), :].astype(int)
                one_hot_encoder.fit(to_fit)
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
        # %% min_max_transformed_col and standard_transformed_col
        for _ in ('min_max_transformed_col', 'standard_transformed_col', 'quantile_transformed_col'):
            if self.transformed_cols_meta[_] != ():
                _index = list(self.transformed_cols_meta[_])
                trans = transformed_cols_args[_].transform(self.data[_index])
                transformed_data.loc[:, _index] = trans

        # %% one_hot_transformed_col
        if self.transformed_cols_meta['one_hot_transformed_col'] != ():
            _ = 'one_hot_transformed_col'
            _index = list(self.transformed_cols_meta[_])
            to_trans = self.data[_index].values
            all_ok_mask = np.all(~np.isnan(to_trans), axis=1)
            to_trans = to_trans[all_ok_mask, :].astype(int)
            trans = transformed_cols_args[_].transform(to_trans)
            transformed_data.loc[all_ok_mask, _index] = trans.toarray().astype(int)

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

    def inverse_transform(self, data_ndarray: ndarray, names: Sequence[Sequence]):
        assert data_ndarray.ndim == 3 and names.__len__() == data_ndarray.shape[2]
        transformed_cols_args, new_multi_index_columns = load_pkl_file(self._transformation_args_file_path)
        data_ndarray_inverse_transformed = np.full((*data_ndarray.shape[:-1], len({x[0] for x in names})), np.nan)

        names_ordered_set = list()
        set_obj = set()
        for name in names:
            if name not in set_obj:
                names_ordered_set.append(name)
                set_obj.add(name)

        for i, this_name in enumerate(names_ordered_set):
            if ("cos" in this_name) or ("sin" in this_name):
                raise NotImplementedError

            if "min_max" in this_name:
                index = self.transformed_cols_meta['min_max_transformed_col'].index(this_name[0])
                scaler_obj = transformed_cols_args['min_max_transformed_col']
            elif "standard" in this_name:
                index = self.transformed_cols_meta['standard_transformed_col'].index(this_name[0])
                scaler_obj = transformed_cols_args['standard_transformed_col']
            elif "quantile" in this_name:
                index = self.transformed_cols_meta['quantile_transformed_col'].index(this_name[0])
                scaler_obj = transformed_cols_args['quantile_transformed_col']
            elif "one_hot" in this_name:
                index = self.transformed_cols_meta['one_hot_transformed_col'].index(this_name[0])
                scaler_obj = transformed_cols_args['one_hot_transformed_col']
            else:
                raise NotImplementedError

            if "one_hot" in this_name:
                for j in range(data_ndarray.shape[0]):
                    data_ndarray_inverse_transformed[j, :, i] = scaler_obj.inverse_transform(
                        data_ndarray[j, :, :][:, [idx for idx, x in enumerate(names) if x == this_name]]
                    ).astype(int)
            else:
                poxy_ndarrays = []
                for j in range(data_ndarray.shape[0]):
                    poxy_ndarray = np.full((data_ndarray.shape[1], scaler_obj.n_features_in_), np.nan)
                    poxy_ndarray[:, index] = data_ndarray[j, :, i]
                    poxy_ndarrays.append(poxy_ndarray)
                poxy_ndarrays = np.array(poxy_ndarrays)
                old_shape = poxy_ndarrays.shape
                temp = scaler_obj.inverse_transform(np.reshape(poxy_ndarrays, (-1, old_shape[-1])))
                data_ndarray_inverse_transformed[:, :, i] = np.reshape(temp, old_shape)[:, :, index]

        return data_ndarray_inverse_transformed

    def _stacked_shift(self, stacked_shift_col, stacked_shift_size, how_many_stacked):
        if (stacked_shift_col or stacked_shift_size or how_many_stacked) == ():
            returned = self.transformed_data
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
            returned = self.transformed_data
        if np.unique(np.diff(returned.index.values)) != 1:
            min_diff = np.min(np.unique(np.diff(self.transformed_data.index.values)))
            msg = f"np.unique(np.diff(returned.index.values))!=1, reindex to {min_diff / np.timedelta64(1, 's')} s"
            warnings.warn(msg)
            returned = returned.reindex(pd.date_range(start=returned.index[0],
                                                      end=returned.index[-1],
                                                      freq=f"{min_diff}N"))
        return returned

    def __getitem__(self, date_time_range: Tuple[datetime.datetime, datetime.datetime]):
        mask = np.bitwise_and(self.data.index >= date_time_range[0], self.data.index < date_time_range[1])
        copied = copy.deepcopy(self)
        copied.data = copied.data[mask]
        copied.transformed_data = copied.transformed_data[mask]
        return copied

    def windowed_dataset(
            self, *,
            x_window_length: datetime.timedelta,
            y_window_length: datetime.timedelta,
            x_y_start_index_diff: datetime.timedelta,
            window_shift: datetime.timedelta,
            batch_drop_remainder=False,
            batch_size: int,
    ):

        # return type: Tuple[tf.data.Dataset, Callable, pd.DataFrame, pd.DataFrame]
        assert np.unique(np.diff(self.transformed_data.index.values)).size == 1
        # Meta info for sliding window
        freq = (self.transformed_data.index.values[1] - self.transformed_data.index.values[0]) / np.timedelta64(1, 's')
        x_window_size = int(x_window_length.total_seconds() / freq)
        y_window_size = int(y_window_length.total_seconds() / freq)
        x_y_start_index_diff_size = int(x_y_start_index_diff.total_seconds() / freq)
        window_shift_size = int(window_shift.total_seconds() / freq)

        # Get col index info
        predictor_cols_index = np.unique(list(
            chain(*[list(self.transformed_data.columns.__getattribute__('get_locs')([x]))
                    for x in self.predictor_cols],
                  *[[i] for i, col in enumerate(self.transformed_data.columns) if 'shift' in col[1]])
        ))
        dependant_cols_index = list(chain(*[list(self.transformed_data.columns.__getattribute__('get_locs')([x]))
                                            for x in self.dependant_cols]))
        dependant_cols_index = [i for i in dependant_cols_index if 'shift' not in self.transformed_data.columns[i][1]]
        predictor_cols_index.sort()
        dependant_cols_index.sort()

        # Make data
        data_x = self.transformed_data.iloc[:, predictor_cols_index]
        data_y = self.transformed_data.iloc[x_y_start_index_diff_size:, dependant_cols_index]

        def data_generator(return_tf_constant: bool = True):
            i = 0
            while True:
                x_slice = slice(i, i + x_window_size)
                y_slice = slice(i, i + y_window_size)
                if x_slice.stop > data_x.shape[0] or y_slice.stop > data_y.shape[0]:
                    break
                x_sliced_data = data_x.values[x_slice]
                y_sliced_data = data_y.values[y_slice]
                if np.any(np.isnan(x_sliced_data)) or np.any(np.isnan(y_sliced_data)):
                    i += window_shift_size
                    continue
                else:
                    if return_tf_constant:
                        yield tf.constant(x_sliced_data, dtype=tf.float32), tf.constant(y_sliced_data, dtype=tf.float32)
                    else:
                        yield data_x[x_slice], data_y[y_slice]
                i += window_shift_size

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(x_window_size, predictor_cols_index.__len__()), dtype=tf.float32),
                tf.TensorSpec(shape=(y_window_size, dependant_cols_index.__len__()), dtype=tf.float32))
        )

        dataset = dataset.batch(batch_size, drop_remainder=batch_drop_remainder)
        dataset = dataset.prefetch(2)
        return dataset, data_generator, data_x, data_y
