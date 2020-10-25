import matlab.engine
from matlab.mlarray import double, int64
import matlab
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from Ploting.fast_plot_Func import series, time_series, hist, scatter
import copy
from project_utils import project_path_
from numpy import ndarray
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
import os
from File_Management.path_and_file_management_Func import try_to_find_folder_path_otherwise_make_one
from Data_Preprocessing import float_eps
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple
from File_Management.path_and_file_management_Func import try_to_find_file
from python_project_common_path_Var import python_project_common_path_
from typing import Tuple, List
import random
import copy
from Time_Processing.datetime_utils import datetime_one_hot_encoder
import torch
import tensorflow as tf


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


class StackedBiLSTM(torch.nn.Module):
    def __init__(self, *,
                 lstm_layer_num: int = 3,
                 input_feature_len: int,
                 output_feature_len: int,
                 hidden_size: int,
                 bidirectional: bool = True,
                 dropout: float = 0.1,
                 sequence_len: int = None):
        super().__init__()
        self.direction = 2 if bidirectional else 1
        self.sequence_len = sequence_len

        self.lstm_layer = torch.nn.LSTM(input_feature_len,
                                        hidden_size,
                                        num_layers=lstm_layer_num,
                                        batch_first=True,
                                        bidirectional=bidirectional,
                                        dropout=dropout)
        self.linear_layer = torch.nn.Linear(hidden_size * self.direction, output_feature_len)
        self.cuda()

    def forward(self, x):
        # 这里r_out shape永远是(seq, batch, output_size)，与网络的batch_first参数无关
        lstm_layer_out, _ = self.lstm_layer(x, None)
        out = self.linear_layer(lstm_layer_out)
        return out

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()


class SimpleLSTM(StackedBiLSTM):
    def __init__(self, input_feature_len: int, hidden_size: int, output_feature_len: int, sequence_len: int = None):
        super().__init__(lstm_layer_num=1,
                         input_feature_len=input_feature_len,
                         output_feature_len=output_feature_len,
                         hidden_size=hidden_size,
                         bidirectional=False,
                         dropout=0,
                         sequence_len=sequence_len)


class GRUEncoder(torch.nn.Module):
    def __init__(self, *,
                 gru_layer_num: int = 1,
                 input_feature_len: int,
                 sequence_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 device='cuda'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.lstm_layer_num = gru_layer_num
        self.direction = 2 if bidirectional else 1

        self.gru_layer = torch.nn.GRU(input_feature_len,
                                      hidden_size,
                                      num_layers=gru_layer_num,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      dropout=dropout)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x):
        gru_output, h_n = self.gru_layer(x)

        # lstm_output sum-reduced by direction
        gru_output = gru_output.view(x.size(0), self.sequence_len, self.direction, self.hidden_size)
        gru_output = gru_output.sum(2)

        # lstm_states sum-reduced by direction
        h_n = h_n.view(self.lstm_layer_num, self.direction, x.size(0), self.hidden_size)
        h_n = h_n.sum(1)

        return gru_output, h_n

    def init_hidden(self, batch_size: int):
        h_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)
        return h_0


class BahdanauAttention(torch.nn.Module):
    def __init__(self, *,
                 hidden_size: int,
                 units: int,
                 device="cuda",
                 mode_source: str = 'custom'):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        self.mode_source = mode_source
        self.W1 = torch.nn.Linear(hidden_size, units)
        self.W2 = torch.nn.Linear(hidden_size, units)
        self.V = torch.nn.Linear(units, 1)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, encoder_output, last_layer_h_n):
        score = self.V(torch.tanh(self.W1(encoder_output) + self.W2(last_layer_h_n.unsqueeze(1))))
        attention_weights = torch.nn.functional.softmax(score, 1)
        if self.mode_source == 'mode_source':
            context_vector = attention_weights * encoder_output
            context_vector = context_vector.sum(1)
        else:
            context_vector = None
        """
        ax = series(context_vector.squeeze().detach().cpu().numpy(), label='context_vector')
        series(last_layer_h_n.squeeze().detach().cpu().numpy(), ax=ax, label='last_layer_h_n')
        series(attention_weights.squeeze().detach().cpu().numpy(), title='attention_weights')
        """
        tt = 1
        return context_vector, attention_weights


class GRUDecoder(torch.nn.Module):
    def __init__(self, *,
                 gru_layer_num: int = 1,
                 decoder_input_feature_len: int,
                 output_feature_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 attention_units: int = 128,
                 mode_source: str = 'custom',
                 device="cuda"):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        else:
            self.mode_source = mode_source
        self.lstm_layer_num = gru_layer_num
        self.direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        self.gru_layer = torch.nn.GRU(decoder_input_feature_len,
                                      hidden_size,
                                      num_layers=gru_layer_num,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      dropout=dropout)

        self.attention = BahdanauAttention(
            hidden_size=hidden_size,
            units=attention_units
        )  # type: BahdanauAttention

        self.out = torch.nn.Linear(hidden_size, output_feature_len)
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, *, y, encoder_output, h_n, this_time_step: int = None):
        # Only use the hidden information from the last layer for attention
        context_vector, attention_weights = self.attention(encoder_output, h_n[-1])
        if self.mode_source == 'nlp':
            y = torch.cat((context_vector.unsqueeze(1), y.unsqueeze(1)), -1)
        ###############################################################################################################
        else:
            attention_weights = (attention_weights - attention_weights.min(1, keepdim=True).values) / (
                    attention_weights.max(1, keepdim=True).values - attention_weights.min(1, keepdim=True).values)
            y = encoder_output * attention_weights
        gru_output, h_n = self.gru_layer(y[:, this_time_step, :].unsqueeze(1), h_n)
        ###############################################################################################################
        # else:
        #     where_max = attention_weights.argmax(1)
        #     y = torch.zeros((encoder_output.size(0), 1, encoder_output.size(2)), device=self.device)
        #     for this_batch_index in range(where_max.size(0)):
        #         y[this_batch_index] = encoder_output[this_batch_index, where_max[this_batch_index], :]
        # gru_output, h_n = self.gru_layer(y, h_n)
        ###############################################################################################################

        output = self.out(gru_output.squeeze(1))
        return output, h_n, attention_weights

    def init_hidden(self, batch_size: int):
        h_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)
        return h_0


class GRUEncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, *, gru_encoder: GRUEncoder,
                 gru_decoder: GRUDecoder,
                 output_sequence_len: int,
                 output_feature_len: int,
                 teacher_forcing: float = 0.001,
                 device="cuda",
                 mode_source: str = 'custom'):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        else:
            self.mode_source = mode_source
        self.gru_encoder = gru_encoder  # type: GRUEncoder
        self.gru_decoder = gru_decoder  # type: GRUDecoder
        self.output_sequence_len = output_sequence_len
        self.output_feature_len = output_feature_len
        self.teacher_forcing = teacher_forcing
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x, y=None):
        encoder_output, encoder_h_n = self.gru_encoder(x)
        # if self.mode_source == 'custom':
        #     decoder_h_n = self.gru_decoder.init_hidden(x.size(0))
        # else:
        decoder_h_n = encoder_h_n
        outputs = torch.zeros((x.size(0), self.output_sequence_len, self.output_feature_len),
                              device=self.device)
        this_time_step_decoder_output = None
        for i in range(self.output_sequence_len):
            if i == 0:
                this_time_step_decoder_input = torch.zeros((x.size(0), self.output_feature_len),
                                                           device=self.device)
            else:
                if (y is not None) and (torch.rand(1) < self.teacher_forcing):
                    this_time_step_decoder_input = y[:, i - 1, :]
                else:
                    this_time_step_decoder_input = this_time_step_decoder_output

            this_time_step_decoder_output, decoder_h_n, _ = self.gru_decoder(
                y=this_time_step_decoder_input,
                encoder_output=encoder_output,
                h_n=decoder_h_n,
                this_time_step=i
            )

            outputs[:, i, :] = this_time_step_decoder_output

        return outputs

    def set_train(self):
        self.gru_encoder.train()
        self.gru_decoder.train()
        self.train()

    def set_eval(self):
        self.gru_encoder.eval()
        self.gru_decoder.eval()
        self.eval()


class TensorFlowLSTMEncoder(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, *, hidden_size: int, training_mode: bool = True):
        super().__init__()
        self.training_mode = training_mode
        self.hidden_size = hidden_size

        self.lstm_layer = tf.keras.layers.LSTM(hidden_size,
                                               return_sequences=True,
                                               return_state=True)

    def call(self, x, h_0_c_0_list):
        lstm_layer_output, h_n, c_n = self.lstm_layer(inputs=x,
                                                      training=self.training_mode,
                                                      initial_state=h_0_c_0_list)
        return lstm_layer_output, [h_n, c_n]

    def initialize_h_0_c_0(self, batch_size: int) -> list:
        return [tf.zeros((batch_size, self.hidden_size)), tf.zeros((batch_size, self.hidden_size))]


class TensorFlowAttention(tf.keras.layers.Layer):
    # ref: https://www.tensorflow.org/tutorials/text/nmt_with_attention
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # TODO: This attention only cares about h_n
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class TensorFlowLSTMDecoder(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, *, hidden_size: int, training_mode: bool = True, output_feature_len: int):
        super().__init__()
        self.training_mode = training_mode
        self.hidden_size = hidden_size

        self.lstm_layer = tf.keras.layers.LSTM(hidden_size,
                                               return_sequences=True,
                                               return_state=True)
        self.fully_connected_layer = tf.keras.layers.Dense(output_feature_len)
        self.attention = TensorFlowAttention(self.hidden_size)

    def call(self, x, h_0_c_0_list, encoder_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(
            query=h_0_c_0_list[0],  # Only h_0 will be used for attention
            values=encoder_output
        )
        # context_vector = h_0_c_0_list[0]
        # attention_weights = None
        # x shape after concatenation == (batch_size, 1, x_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(x, 1)],
                      axis=-1)

        # passing the concatenated vector to the LSTM
        lstm_layer_output, h_n, c_n = self.lstm_layer(x,
                                                      training=self.training_mode)
        # output shape == (batch_size, output_feature_len)
        x = self.fully_connected_layer(lstm_layer_output)
        return x, [h_n, c_n], attention_weights


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
