import pandas as pd
from typing import Tuple, List, Callable
import torch
import tensorflow as tf
from Ploting.fast_plot_Func import *
import tensorflow_probability as tfp
import edward2 as ed
from abc import abstractmethod, ABCMeta

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")

tf.keras.backend.set_floatx('float32')


class BayesianConv1DBiLSTM(metaclass=ABCMeta):
    __slots__ = ("input_shape", "output_shape", "batch_size",
                 "conv1d_hypers", "conv1d_layer_count", "use_encoder_decoder",
                 "maxpool1d_hypers",
                 "bilstm_hypers", "bilstm_layer_count",
                 "dense_hypers")

    def __init__(self, *,
                 input_shape: Sequence,
                 output_shape: Sequence,
                 batch_size: int,
                 conv1d_layer_count: int = 1,
                 bilstm_layer_count: int = 1,
                 use_encoder_decoder: bool,
                 dense_hypers_units: int,
                 **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.conv1d_layer_count = conv1d_layer_count
        self.bilstm_layer_count = bilstm_layer_count
        self.use_encoder_decoder = use_encoder_decoder

        # %% Bayesian Conv1D hyper parameters and their default values
        self.conv1d_hypers = {
            "filters": kwargs.get("conv1d_hypers_filters", 8),
            "kernel_size": kwargs.get("conv1d_hypers_kernel_size", 5),
            "activation": kwargs.get("conv1d_hypers_activation", "relu"),
            "padding": kwargs.get("conv1d_hypers_padding", "valid"),
            "input_shape": self.input_shape,
            "kernel_prior_fn": kwargs.get("conv1d_hypers_kernel_prior_fn", self.get_conv1d_kernel_prior_fn()),
            "kernel_posterior_fn": kwargs.get("conv1d_hypers_kernel_posterior_fn",
                                              self.get_conv1d_kernel_posterior_fn()),
            "kernel_divergence_fn": kwargs.get("conv1d_hypers_kernel_divergence_fn",
                                               self.get_conv1d_kernel_divergence_fn()),
            "bias_prior_fn": kwargs.get("conv1d_hypers_bias_prior_fn", self.get_conv1d_bias_prior_fn()),
            "bias_posterior_fn": kwargs.get("conv1d_hypers_bias_posterior_fn",
                                            self.get_conv1d_bias_posterior_fn()),
            "bias_divergence_fn": kwargs.get("conv1d_hypers_bias_divergence_fn",
                                             self.get_conv1d_bias_divergence_fn())
        }

        # %% Bayesian MaxPooling hyper parameters and their default values
        self.maxpool1d_hypers = {
            "pool_size": kwargs.get("maxpool1d_hypers_pool_size", 5),
            "padding": kwargs.get("maxpool1d_hypers_padding", "valid")
        }

        # %% Bayesian BiLSTM hyper parameters and their default values
        self.bilstm_hypers = {
            "units": kwargs.get("bilstm_hypers_units", 32),
        }

        # %% Bayesian Dense hyper parameters and their default values
        self.dense_hypers = {
            "units": dense_hypers_units,
            "make_prior_fn": kwargs.get("dense_hypers_make_prior_fn", self.get_dense_make_prior_fn()),
            "make_posterior_fn": kwargs.get("dense_hypers_make_posterior_fn", self.get_dense_make_posterior_fn()),
            "kl_weight": kwargs.get("dense_hypers_kl_weight", self.kl_weight),
        }

    def build(self):
        layers = [self.get_convolution1d_reparameterization_layer() for _ in range(self.conv1d_layer_count)]
        # layers = [tf.keras.layers.LocallyConnected1D(filters=8, kernel_size=5, input_shape=self.input_shape)]
        layers.append(tf.keras.layers.MaxPooling1D(**self.maxpool1d_hypers)),
        if self.bilstm_layer_count == 0:
            layers.append(tf.keras.layers.Flatten())
        else:
            layers.extend([self.get_bilstm_reparameterization_layer(True if i < self.bilstm_layer_count - 1 else False)
                           for i in range(self.bilstm_layer_count)])
            layers.append(tf.keras.layers.RepeatVector(self.output_shape[0]))
            if self.use_encoder_decoder:
                layers.append(self.get_bilstm_reparameterization_layer(True))
        layers.append(self.get_dense_variational_layer())
        layers.append(self.get_distribution_layer())
        model = tf.keras.Sequential(layers)
        return model

    # Function to define the spike and slab distribution
    @staticmethod
    def spike_and_slab(event_shape, dtype):
        distribution = tfd.Mixture(
            cat=tfd.Categorical(probs=tf.cast([0.5, 0.5], dtype=dtype)),
            components=[
                tfd.Independent(tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=1.0 * tf.ones(event_shape, dtype=dtype)),
                    reinterpreted_batch_ndims=1),
                tfd.Independent(tfd.Normal(
                    loc=tf.zeros(event_shape, dtype=dtype),
                    scale=10.0 * tf.ones(event_shape, dtype=dtype)),
                    reinterpreted_batch_ndims=1)],
            name='spike_and_slab')
        return distribution

    @property
    def kl_weight(self):
        return 1 / self.batch_size

    @staticmethod
    def get_conv1d_kernel_prior_fn():
        return tfpl.default_multivariate_normal_fn

    @staticmethod
    def get_conv1d_kernel_posterior_fn():
        return tfpl.default_mean_field_normal_fn(is_singular=False)

    def get_conv1d_kernel_divergence_fn(self):
        return lambda q, p, _: tfd.kl_divergence(q, p) * self.kl_weight

    @staticmethod
    def get_conv1d_bias_prior_fn():
        return tfpl.default_multivariate_normal_fn

    @staticmethod
    def get_conv1d_bias_posterior_fn():
        return tfpl.default_mean_field_normal_fn(is_singular=False)

    def get_conv1d_bias_divergence_fn(self):
        return lambda q, p, _: tfd.kl_divergence(q, p) * self.kl_weight

    def get_convolution1d_reparameterization_layer(self) -> tfpl.Convolution1DReparameterization:
        assert self
        interesting_hypers = ("filters", "kernel_size", "activation", "padding", "input_shape",
                              "kernel_prior_fn", "kernel_posterior_fn", "kernel_divergence_fn",
                              "bias_prior_fn", "bias_posterior_fn", "bias_divergence_fn")
        args_source_code = """""".join([f'{x} = self.conv1d_hypers["{x}"],\n' for x in interesting_hypers])
        conv1d_layer = eval(f"tfpl.Convolution1DReparameterization({args_source_code})")
        return conv1d_layer

    def get_bilstm_kernel_regularizer(self):
        return ed.tensorflow.regularizers.NormalKLDivergence(mean=0., stddev=1., scale_factor=self.kl_weight)

    def get_bilstm_recurrent_regularizer(self):
        return ed.tensorflow.regularizers.NormalKLDivergence(mean=0., stddev=1., scale_factor=self.kl_weight)

    def get_bilstm_reparameterization_layer(self, return_sequences):
        bilstm_cell = ed.layers.LSTMCellReparameterization(**self.bilstm_hypers,
                                                           kernel_regularizer=self.get_bilstm_kernel_regularizer(),
                                                           recurrent_regularizer=self.get_bilstm_kernel_regularizer())
        bilstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(bilstm_cell,
                                                                         return_sequences=return_sequences))
        return bilstm_layer

    def get_dense_make_prior_fn(self):
        def dense_make_prior_fn(kernel_size, bias_size, dtype=tf.float32):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential([
                tfpl.DistributionLambda(
                    lambda t: self.spike_and_slab(n, dtype)
                )
            ])
            return prior_model

        return dense_make_prior_fn

    @staticmethod
    def get_dense_make_posterior_fn():
        def dense_make_posterior_fn(kernel_size, bias_size, dtype=tf.float32):
            n = kernel_size + bias_size
            posterior_model = tf.keras.Sequential([
                tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
                tfpl.IndependentNormal(n)
            ])
            return posterior_model

        return dense_make_posterior_fn

    def get_dense_variational_layer(self):
        assert self
        interesting_hypers = ("units", "make_prior_fn", "make_posterior_fn", "kl_weight")
        args_source_code = """""".join([f'{x} = self.dense_hypers["{x}"],\n' for x in interesting_hypers])
        dense_layer = eval(f"tfpl.DenseVariational({args_source_code})")
        return dense_layer

    @abstractmethod
    def get_distribution_layer(self, dtype=tf.float32):
        # dist_layer = tfpl.DistributionLambda(
        #     make_distribution_fn=lambda t: tfd.Independent(
        #         tfd.Mixture(
        #             cat=tfd.Categorical(probs=tf.cast([[1 / 3, 1 / 3, 1 / 3] for i in range(6)], dtype=dtype)),
        #             components=[tfd.Independent(tfd.Normal(loc=tf.zeros(1*6, dtype=dtype),
        #                                                    scale=1.0 * tf.ones(1*6, dtype=dtype)),
        #                                         reinterpreted_batch_ndims=1)
        #                         for j in range(3)]
        #         ),
        #         reinterpreted_batch_ndims=1
        #     ),
        #     convert_to_tensor_fn=lambda s: s.sample()
        # )

        # dist_layer = tfpl.MixtureSameFamily(num_components=self.dist_hypers["num_components"],
        #                                     component_layer=tfpl.MultivariateNormalTriL(self.output_shape[1]),
        #                                     convert_to_tensor_fn=tfd.Distribution.sample)

        # dist_layer = tfpl.IndependentNormal(event_shape=self.output_shape,
        #                                     convert_to_tensor_fn=tfd.Distribution.sample)

        # dist_layer = tfpl.MultivariateNormalTriL(self.output_shape[0] * self.output_shape[1])
        # dist_layer = tfpl.MixtureNormal(num_components=self.dist_hypers["num_components"],
        #                                 event_shape=self.output_shape)
        # return dist_layer
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


class TensorFlowCovBiLSTMEncoder(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, *,
                 training_mode: bool = True,
                 conv1d_layer_filters: int = 32,
                 lstm_layer_hidden_size: int):
        super().__init__()
        self.training_mode = training_mode
        # Convolutional 1D
        self.conv1d_layer = tf.keras.layers.Conv1D(filters=conv1d_layer_filters, kernel_size=3,
                                                   strides=1, padding="causal",
                                                   activation="relu",
                                                   dtype='float32')
        # %% Bidirectional LSTM layer
        # TODO 改成BiLSTM，改成多层
        self.lstm_layer_hidden_size = lstm_layer_hidden_size
        self.lstm_layer = tf.keras.layers.LSTM(lstm_layer_hidden_size,
                                               return_sequences=True,
                                               return_state=True,
                                               dtype='float32')

    def call(self, x=None, h_0_c_0_list=None, **kwargs):
        conv1d_layer_output = self.conv1d_layer(inputs=x,
                                                training=self.training_mode)
        lstm_layer_output, h_n, c_n = self.lstm_layer(inputs=conv1d_layer_output,
                                                      training=self.training_mode,
                                                      initial_state=h_0_c_0_list)
        return lstm_layer_output, [h_n, c_n]

    def initialize_h_0_c_0(self, batch_size: int) -> list:
        return [tf.zeros((batch_size, self.lstm_layer_hidden_size), dtype='float32'),
                tf.zeros((batch_size, self.lstm_layer_hidden_size), dtype='float32')]


class TensorFlowBahdanauAttention(tf.keras.layers.Layer):
    # ref: https://www.tensorflow.org/tutorials/text/nmt_with_attention
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, dtype='float32')
        self.W2 = tf.keras.layers.Dense(units, dtype='float32')
        self.V = tf.keras.layers.Dense(1, dtype='float32')

    def call(self, query=None, values=None, **kwargs):
        # https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms#
        #  query is our decoder_states and value is our encoder_outputs.

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

    def __init__(self, *, training_mode: bool = True,
                 lstm_layer_hidden_size: int,
                 output_feature_len: int):
        super().__init__()
        self.training_mode = training_mode
        self.lstm_layer_hidden_size = lstm_layer_hidden_size

        self.lstm_layer = tf.keras.layers.LSTM(lstm_layer_hidden_size,
                                               return_sequences=True,
                                               return_state=True,
                                               dtype='float32')
        self.fully_connected_layer = tf.keras.layers.Dense(output_feature_len, dtype='float32')
        self.attention = TensorFlowBahdanauAttention(self.lstm_layer_hidden_size)

    def call(self, x=None, h_0_c_0_list=None, encoder_output=None, **kwargs):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(
            query=h_0_c_0_list[0],  # Only h_0 will be used for attention
            values=encoder_output
        )
        # context_vector = h_0_c_0_list[0]
        # attention_weights = None
        # x shape after concatenation == (batch_size, 1, x_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x],
                      axis=-1)

        # passing the concatenated vector to the LSTM
        lstm_layer_output, h_n, c_n = self.lstm_layer(x,
                                                      training=self.training_mode)
        # output shape == (batch_size, output_feature_len)
        x = self.fully_connected_layer(lstm_layer_output,
                                       training=self.training_mode)
        return x, [h_n, c_n], attention_weights


class GradientsAnalyser:
    __slots__ = ("gradients", "predictor_names")

    def __init__(self, gradients: ndarray, predictor_names: List[str]):
        assert gradients.ndim == 3

        self.gradients = gradients
        self.predictor_names = predictor_names

    def aggregate_over_cos_sin_and_one_hot(self, apply_abs_op: bool = True) -> Tuple[ndarray, List[str]]:
        multi_index = pd.MultiIndex.from_tuples(self.predictor_names)
        new_predictor_names = multi_index.get_level_values(0).drop_duplicates()
        new_gradients = np.full((*self.gradients.shape[:-1], new_predictor_names.__len__()), np.nan)
        for i, this_gradient in enumerate(self.gradients):
            this_gradient_df = pd.DataFrame(data=this_gradient, columns=multi_index)
            for j, this_new_predictor_name in enumerate(new_predictor_names):
                df = this_gradient_df[this_new_predictor_name]
                new_gradients[i, :, j] = np.mean(np.abs(df.values) if apply_abs_op else df.values, axis=1)

        return new_gradients, new_predictor_names.tolist()

    def aggregate_over_all_samples(self, apply_abs_op: bool = True) -> Tuple[ndarray, List[str]]:
        new_gradients, new_predictor_names = self.aggregate_over_cos_sin_and_one_hot(apply_abs_op=apply_abs_op)
        new_gradients = np.mean(new_gradients, axis=0)
        return new_gradients, new_predictor_names

    @staticmethod
    def plot(gradients, predictor_names, *, plot_total_steps: bool = True,
             plot_individual_steps: bool = False, log_scale: bool = False):
        if gradients.ndim == 2:
            gradients = gradients[np.newaxis, ...]
        if plot_total_steps:
            for this_gradient in gradients:
                y = np.mean(this_gradient, 0)
                if log_scale:
                    y = np.log10(y)
                stem(predictor_names, y, x_ticks_rotation=90)
        if plot_individual_steps:
            ax = None
            for this_gradient in gradients:
                this_gradient = this_gradient.T
                for i, this_dim in enumerate(this_gradient):
                    ax = series(this_dim, ax=ax, label=predictor_names[i])
