import matlab.engine
from matlab.mlarray import double, int64
import matlab
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from Ploting.fast_plot_Func import series, time_series, hist, scatter
import copy
from project_path_Var import project_path_
from numpy import ndarray
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
import os
from File_Management.path_and_file_management_Func import try_to_find_folder_path_otherwise_make_one
from Data_Preprocessing import float_eps
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple
from File_Management.path_and_file_management_Func import try_to_find_file
from python_project_common_path_Var import python_project_common_path_
from typing import Tuple
import random
import copy
from Time_Processing.datetime_utils import datetime_one_hot_encoder
import torch
from torch import nn
from torch.autograd import Variable


def use_min_max_scaler_and_save(data_to_be_normalised: ndarray, file_: str, **kwargs):
    """
    将MinMaxScaler作用于training set。并将min和max储存到self.results_path
    :return:
    """

    @load_exist_pkl_file_otherwise_run_and_save(file_)
    def data_scaling():
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler

    scaled_data = data_scaling.fit_transform(data_to_be_normalised)
    return scaled_data


def shift_and_concatenate_data(data_to_be_sc, shift: int, **kwargs):
    """

    :param data_to_be_sc:
    :param shift: 负数代表往past移动
    :return:
    """
    dims = data_to_be_sc.shape[1]
    results = np.full((data_to_be_sc.shape[0], abs(shift) * data_to_be_sc.shape[1]), np.nan)
    for i in range(abs(shift)):
        results[:, i * dims:(i + 1) * dims] = np.roll(data_to_be_sc, int(i * shift / shift), axis=0)
    # results[:abs(shift)] = np.nan

    results_final = np.full((data_to_be_sc.shape[0], abs(shift) * data_to_be_sc.shape[1]), np.nan)
    for i in range(dims):
        results_final[:, i * abs(shift):(i + 1) * abs(shift)] = results[:, -1 - i::-dims]

    return results_final


def get_training_mask_in_train_validation_set(train_validation_set_len: int, validation_pct: float):
    training_mask = np.full((train_validation_set_len,), True)
    training_mask[random.choices(range(training_mask.__len__()),
                                 k=int(training_mask.__len__() * validation_pct))] = False
    return training_mask


def prepare_data_for_nn(*, datetime_: ndarray = None, x: ndarray, y: ndarray,
                        validation_pct: float,
                        x_time_step: int = None,
                        y_time_step: int = None, path_: str,
                        **kwargs) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    准备数据给nn训练
    :param datetime_: 时间特征
    :param x:
    :param y:
    :param validation_pct:
    :param x_time_step:
    :param y_time_step:
    :param path_:
    :return:
    """
    x, y = copy.deepcopy(x), copy.deepcopy(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # scaler
    x_training_validation_set_core = use_min_max_scaler_and_save(x, file_=path_ + 'x_scaler.pkl')
    y_training_validation_set_core = use_min_max_scaler_and_save(y, file_=path_ + 'y_scaler.pkl')
    # shift_and_concatenate_data
    if (x_time_step is not None) and (y_time_step is not None):
        x_training_validation_set_core = shift_and_concatenate_data(x_training_validation_set_core, -1 * x_time_step)
        y_training_validation_set_core = np.roll(y_training_validation_set_core, -y_time_step, axis=0)
        y_training_validation_set_core = shift_and_concatenate_data(y_training_validation_set_core, y_time_step)
    # 包含时间数据
    if datetime_ is not None:
        datetime_ = datetime_one_hot_encoder(datetime_, **kwargs)
        x_train_validation = np.concatenate((datetime_,
                                             x_training_validation_set_core),
                                            axis=1)
    else:
        x_train_validation = x_training_validation_set_core
    y_train_validation = y_training_validation_set_core

    if (x_time_step is not None) and (y_time_step is not None):
        # 截断无用数据
        x_train_validation = x_train_validation[x_time_step:]
        y_train_validation = y_train_validation[x_time_step:]
        # 非连续的选取
        # x_train_validation = x_train_validation[::y_time_step]
        # y_train_validation = y_train_validation[::y_time_step]

    # 依据validation_pct，选取train和validation
    training_mask = get_training_mask_in_train_validation_set(x_train_validation.shape[0], validation_pct)

    # 划分train和validation
    x_train = x_train_validation[training_mask, :]
    y_train = y_train_validation[training_mask, :]
    x_validation = x_train_validation[~training_mask, :]
    y_validation = y_train_validation[~training_mask, :]
    return x_train, y_train, x_validation, y_validation


class BayesCOVN1DLSTM:
    pass


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, output_size)
        self.cuda()

    def forward(self, x):
        # 这里r_out shape永远是(seq, batch, output_size)，与网络的batch_first参数无关
        r_out, (h_n, h_c) = self.lstm_layer(x, None)
        # 如果网络的batch_first = True，这里out的shape就是(batch, seq, output_size)
        # 否则，这里out的shape就是(seq, batch, output_size)
        out = self.linear_layer(r_out)
        return out

    def init_hidden(self, x):
        h_0 = torch.zeros(1, x.size[1], self.hidden_size, device='cuda:0')
        c_0 = torch.zeros(1, x.size[1], self.hidden_size, device='cuda:0')
        return h_0, c_0


class StackedBiLSTM(SimpleLSTM):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, dropout: float):
        super().__init__(input_size, hidden_size, output_size)
        self.lstm_layer = nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers=3,
                                  batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size * 2, output_size)
        self.cuda()


class MatlabLSTM:
    __slots__ = ('lstm_file_',)

    def __init__(self, lstm_file_: str):
        self.lstm_file_ = lstm_file_

    def train(self, x_train, y_train, x_validation, y_validation, max_epochs):
        if not try_to_find_file(self.lstm_file_):
            eng = matlab.engine.start_matlab()
            eng.addpath(python_project_common_path_ + r'Regression_Analysis\LSTM_MATLAB', nargout=0)
            eng.train_LSTM_and_save(double(x_train.tolist()),
                                    double(y_train.tolist()),
                                    double(x_validation.tolist()),
                                    double(y_validation.tolist()),
                                    self.lstm_file_,
                                    max_epochs,
                                    nargout=0)
            eng.quit()

    def test(self, x_test):
        if not try_to_find_file(self.lstm_file_):
            raise Exception("没有找到训练好的模型: {}".format(self.lstm_file_))
        eng = matlab.engine.start_matlab()
        eng.addpath(python_project_common_path_ + r'Regression_Analysis\LSTM_MATLAB', nargout=0)
        result = eng.load_LSTM_and_predict(double(x_test.tolist()),
                                           self.lstm_file_,
                                           nargout=1)
        eng.quit()
        result = np.asarray(result)
        return result
