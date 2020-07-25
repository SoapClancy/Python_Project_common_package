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
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device='cuda:0'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear_layer_out = nn.Linear(hidden_size, output_size)
        if 'cuda' in device:
            self.cuda()

    def forward(self, x):
        # 这里r_out shape永远是(seq, batch, output_size)，与网络的batch_first参数无关
        r_out, (h_n, h_c) = self.lstm_layer(x, None)
        # 如果网络的batch_first = True，这里out的shape就是(batch, seq, output_size)
        # 否则，这里out的shape就是(seq, batch, output_size)
        out = self.linear_layer_out(r_out)
        return out

    def init_hidden(self, x):
        h_0 = torch.zeros(1, x.size[1], self.hidden_size, device='cuda:0')
        c_0 = torch.zeros(1, x.size[1], self.hidden_size, device='cuda:0')
        return h_0, c_0


class LSTMEncoder(nn.Module):
    def __init__(self, *, lstm_num_layers: int = 1,
                 input_feature_len: int,
                 output_feature_len: int,
                 sequence_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.output_feature_len = output_feature_len
        self.num_layers = lstm_num_layers
        self.direction = 2 if bidirectional else 1

        self.lstm_layer = nn.LSTM(input_feature_len,
                                  hidden_size,
                                  num_layers=lstm_num_layers,
                                  batch_first=True,
                                  bidirectional=bidirectional,
                                  dropout=dropout)

        self.decoder_first_input_dense = nn.Linear(hidden_size, output_feature_len)

        self.cuda()

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm_layer(x)

        # lstm_output sum-reduced by direction
        lstm_output = lstm_output.view(x.size(0), self.sequence_len, self.direction, self.hidden_size)
        lstm_output = lstm_output.sum(2)

        # lstm_states sum-reduced by direction
        h_n = h_n.view(self.num_layers, self.direction, x.size(0), self.hidden_size)
        c_n = c_n.view(self.num_layers, self.direction, x.size(0), self.hidden_size)

        # Only use the information from the last layer
        h_n, c_n = h_n[-1], c_n[-1]
        h_n, c_n = h_n.sum(0), c_n.sum(0)

        return lstm_output, (h_n, c_n), self.decoder_first_input_dense(h_n)

    def init_hidden(self, x):
        h_0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size, device='cuda:0')
        c_0 = torch.zeros(self.num_layers * self.direction, x.size(0), self.hidden_size, device='cuda:0')

        return h_0, c_0


class LSTMCellDecoder(nn.Module):
    def __init__(self, *,
                 output_feature_len: int,
                 hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size=output_feature_len,
            hidden_size=hidden_size
        )
        # 我觉得应该hidden_size*2，因为lstm cell输出有两个状态h_n, c_n
        self.out = nn.Linear(hidden_size, output_feature_len)
        # TODO attention
        self.attention = False
        self.dropout = nn.Dropout(dropout)
        self.cuda()

    def forward(self, y, prev_h_n_and_c_n_tuple: tuple) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_n, c_n = self.lstm_cell(y, prev_h_n_and_c_n_tuple)
        output = self.out(h_n)
        h_n_c_n_cat = self.dropout(torch.cat((h_n, c_n), 1))
        return output, (h_n_c_n_cat[:, :int(h_n_c_n_cat.size(1) / 2)], h_n_c_n_cat[:, int(h_n_c_n_cat.size(1) / 2):])


class LSTMEncoderDecoderWrapper(nn.Module):
    def __init__(self, *, lstm_encoder: LSTMEncoder,
                 lstm_decoder_cell: LSTMCellDecoder,
                 teacher_forcing: float = 0.3,
                 output_sequence_len: int,
                 output_feature_len: int,
                 decoder_input=True):
        super().__init__()
        self.lstm_encoder = lstm_encoder  # type: LSTMEncoder
        self.lstm_decoder_cell = lstm_decoder_cell  # type: LSTMCellDecoder
        self.teacher_forcing = teacher_forcing
        self.output_sequence_len = output_sequence_len
        self.output_feature_len = output_feature_len
        self.decoder_input = decoder_input
        self.cuda()

    def forward(self, x, y=None):
        outputs = torch.zeros(x.size(0), self.output_sequence_len, self.output_feature_len, device='cuda:0')
        _, (encoder_h_n, encoder_c_n), decoder_first_input = self.lstm_encoder(x)
        prev_h_n_and_c_n_tuple = (encoder_h_n, encoder_c_n)
        lstm_decoder_cell_output = None
        for i in range(self.output_sequence_len):
            if i == 0:
                step_decoder_input = decoder_first_input
            else:
                if (y is not None) and (torch.rand(1).item() < self.teacher_forcing):
                    step_decoder_input = y[:, i - 1, :]
                else:
                    step_decoder_input = lstm_decoder_cell_output

            lstm_decoder_cell_output, prev_h_n_and_c_n_tuple = self.lstm_decoder_cell(step_decoder_input,
                                                                                      prev_h_n_and_c_n_tuple)
            outputs[:, i, :] = lstm_decoder_cell_output
        return outputs


class StackedBiLSTM(SimpleLSTM):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, *,
                 dropout: float, device='cuda:0', sequence_len: int = None):
        super().__init__(input_size, hidden_size, output_size, device=device)
        self.direction = 2
        self.num_layers = 3
        self.sequence_len = sequence_len

        # self.linear_layer_in = nn.Linear(input_size, output_size)
        self.lstm_layer = nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers=3,
                                  batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout)
        self.linear_layer_out = nn.Linear(hidden_size * 2, output_size)
        if 'cuda' in device:
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
