from numpy import ndarray
from sklearn.cluster import KMeans
from File_Management.load_save_Func import *
from File_Management.path_and_file_management_Func import *
from typing import Tuple
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, OneDimensionNdarray
import edward2 as ed
from abc import ABCMeta, abstractmethod
from Ploting.fast_plot_Func import *
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import scipy
from Data_Preprocessing.float_precision_control_Func import float_eps
import tensorflow_addons as tfa
import re


tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")
tfp_util = eval("tfp.util")
tfp_math = eval("tfp.math")


class MixtureSameFamilyMeta(type):
    @staticmethod
    def _make_essential_wrapper_method_source_code(clsdict):
        essential_wrapper_method_name = ['prob', 'log_prob', 'cdf', 'log_cdf', 'sample']
        source_code_list = []

        for method_name in essential_wrapper_method_name:
            if method_name not in clsdict:
                source_code = f"""\
                def {method_name}(self, *args, **kwargs):
                    return self.tfd_obj.{method_name}(*args, **kwargs)\
                """
                indent = re.search("def", source_code.split("\n")[0]).regs[0][0]
                source_code = "\n".join([x[indent:] for x in source_code.split("\n")])
                source_code_list.append(source_code)

        return source_code_list

    def __new__(mcs, name, bases, clsdict):
        if name != "MixtureSameFamilyWrapper":
            essential_wrapper_method_source_code = mcs._make_essential_wrapper_method_source_code(clsdict)
            for source_code in essential_wrapper_method_source_code:
                exec(source_code, globals(), clsdict)  # new

        clsobj = super().__new__(mcs, name, bases, dict(clsdict))
        return clsobj


class MixtureSameFamilyWrapper(metaclass=MixtureSameFamilyMeta):
    __slots__ = ('data', 'fit_results_path', 'quantile_meta', 'tfd_obj')

    def __init__(self, *, data=None, fit_results_path: Path = None, quantile_meta: dict):
        assert (data is not None) or (fit_results_path is not None)

        self.data = data
        self.fit_results_path = fit_results_path
        self.quantile_meta = quantile_meta

        self.tfd_obj = None  # type: tfd.Distribution

    def _initial_guess_by_naive_gmm(self, data_bijector: tfb.Bijector = None):
        lowest_bic = np.infty
        n_components_range = range(1, 7)
        best_gmm = None

        data_to_fit = self.data
        if data_bijector is not None:
            data_to_fit = data_bijector.forward(data_to_fit).numpy()
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.fit(data_to_fit)
            now_bic = gmm.bic(data_to_fit)
            if now_bic < lowest_bic:
                lowest_bic = now_bic
                best_gmm = gmm
        ans = {
            "weights": best_gmm.weights_.astype(np.float32),
            "means": best_gmm.means_.astype(np.float32),
            "covariances": best_gmm.covariances_.astype(np.float32)
        }
        return ans

    @abstractmethod
    def _get_dist_trainable(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, *,
            epochs: int = 5000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.01),
            **kwargs):
        assert self.data.shape.__len__() >= 2, "The rank of data should be at least 2"
        assert isinstance(self.fit_results_path, Path)

        dist_trainable = self._get_dist_trainable(**kwargs)

        # Define nll
        def nll_func(_data, nll_dist):
            nll_value = nll_dist.log_prob(_data)
            return -tf.math.reduce_mean(nll_value)
            # return -tf.math.reduce_mean(nll_value[~tf.math.is_nan(nll_value)])

        # Define loss and gradients
        @tf.function
        def get_loss_and_grads(_data, _dist):
            with tf.GradientTape() as tape:
                tape.watch(_dist.trainable_variables)
                _loss = nll_func(_data, _dist)
            _grads = tape.gradient(_loss, _dist.trainable_variables)
            return _loss, _grads

        # Train
        print(f"Ready to fit {dist_trainable.name}")
        for i in range(epochs):
            loss, grads = get_loss_and_grads(self.data, dist_trainable)
            optimiser.apply_gradients(zip(grads, dist_trainable.trainable_variables))
            if i % 100 == 0:
                print(f"epochs = {i}, loss = {loss}")
        return dist_trainable

    def load(self):
        constructor_params = load_pkl_file(self.fit_results_path)

        def params_source_code_maker(dict_obj_gen: str) -> str:
            nonlocal constructor_params
            _params_source_code = []
            for key in eval(dict_obj_gen, globals(), locals()):
                _params_source_code.append(f"""{key}={dict_obj_gen}['{key}'],\n""")
            return ''.join(_params_source_code)

        dist_source_code = f"""\
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.{constructor_params['mixture_distribution']['name']}(
                {params_source_code_maker("constructor_params['mixture_distribution']['params']")}
            ),
            components_distribution=tfd.{constructor_params['components_distribution']['name']}(
                {params_source_code_maker("constructor_params['components_distribution']['params']")}
            )
        )\
        """
        dist = eval(dist_source_code)
        self.tfd_obj = dist

    def quantile(self, value):
        assert 'look_up_tbl_path' in self.quantile_meta

        @load_exist_pkl_file_otherwise_run_and_save(self.quantile_meta['look_up_tbl_path'])
        def get_look_up_tbl():
            assert 'cdf_x_min' in self.quantile_meta
            assert 'cdf_x_max' in self.quantile_meta
            assert 'cdf_x_num' in self.quantile_meta

            _x = np.linspace(self.quantile_meta['cdf_x_min'],
                             self.quantile_meta['cdf_x_max'],
                             self.quantile_meta['cdf_x_num'])
            _y = self.__getattribute__('cdf')(_x)
            return {'cdf_x': _x, 'cdf_y': _y,
                    **{key: self.quantile_meta[key] for key in ['cdf_x_min', 'cdf_x_max', 'cdf_x_num']}}

        look_up_tbl = get_look_up_tbl()
        ans = tfp_math.interp_regular_1d_grid(value,
                                              look_up_tbl['cdf_x_min'],
                                              look_up_tbl['cdf_x_max'],
                                              look_up_tbl['cdf_y']).numpy()
        return ans


class MixtureGaussian(MixtureSameFamilyWrapper):

    def _get_dist_trainable(self, **kwargs):
        initial_guess = self._initial_guess_by_naive_gmm()
        dist_trainable = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tfb.Sigmoid().inverse(initial_guess['weights']))
            ),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=tf.Variable(initial_guess['means']),
                scale_tril=tfp_util.TransformedVariable(tf.linalg.cholesky(initial_guess['covariances']),
                                                        bijector=tfb.FillScaleTriL())
            ),
            name=f'learnable_MixtureGaussian_obj'
        )
        return dist_trainable

    def fit(self, *,
            epochs: int = 1000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.01),
            **kwargs):
        fit_results = super().fit(epochs=epochs, optimiser=optimiser, **kwargs)

        constructor_params = {
            'mixture_distribution': {
                'name': 'Categorical',
                'params': {
                    'logits': fit_results.mixture_distribution.logits.numpy()
                }
            },
            'components_distribution': {
                'name': 'MultivariateNormalTriL',
                'params': {
                    'loc': fit_results.components_distribution.loc.numpy(),
                    'scale_tril': fit_results.components_distribution.scale_tril.numpy()
                }
            }
        }
        save_pkl_file(self.fit_results_path, constructor_params)
        return constructor_params

    def cdf(self, value):
        # build scipy version, since tfp does not have implementation for MultivariateNormalTriL CDF
        weights = tfb.Sigmoid()(self.tfd_obj.mixture_distribution.logits).numpy()
        mean = self.tfd_obj.components_distribution.loc.numpy()
        cov = self.tfd_obj.components_distribution.covariance().numpy()

        obj_maker = eval("scipy.stats.multivariate_normal")
        mixture = []
        for i, now_weight in enumerate(weights):
            mixture.append([obj_maker(mean=mean[i], cov=cov[i]), now_weight])

        ans = np.full(value.shape, 0.)
        for i, ele in enumerate(mixture):
            ans += ele[0].cdf(value) * ele[1]
        ans = np.clip(ans, float_eps, 1. - float_eps)
        return ans


class MixtureTruncatedNormal(MixtureSameFamilyWrapper):
    """
    This is a scalar dist
    """

    def _get_dist_trainable(self, **kwargs):
        initial_guess = self._initial_guess_by_naive_gmm()
        dist_trainable = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tfb.Sigmoid().inverse(initial_guess['weights']))
            ),
            components_distribution=tfd.TruncatedNormal(
                loc=tf.Variable(initial_guess['means'].flatten()),

                scale=tfp_util.TransformedVariable(np.sqrt(initial_guess['covariances'].flatten()),
                                                   bijector=tfb.Softplus()),
                low=kwargs['low'],
                high=kwargs['high']
            ),
            name=f'learnable_TruncatedNormal_obj'
        )
        return dist_trainable

    def fit(self, *,
            epochs: int = 5000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.01),
            **kwargs):
        fit_results = super().fit(epochs=epochs, optimiser=optimiser, **kwargs)

        constructor_params = {
            'mixture_distribution': {
                'name': 'Categorical',
                'params': {
                    'logits': fit_results.mixture_distribution.logits.numpy()
                }
            },
            'components_distribution': {
                'name': 'TruncatedNormal',
                'params': {
                    'loc': fit_results.components_distribution.loc.numpy(),
                    'scale': fit_results.components_distribution.scale.numpy(),
                    'low': fit_results.components_distribution.low.numpy(),
                    'high': fit_results.components_distribution.high.numpy()
                }
            }
        }
        save_pkl_file(self.fit_results_path, constructor_params)
        return fit_results


class MixtureLogitNormal(MixtureSameFamilyWrapper):
    """
    This is a scalar dist
    """

    def _get_dist_trainable(self, **kwargs):
        initial_guess = self._initial_guess_by_naive_gmm(tfb.Invert(tfb.Sigmoid()))

        dist_trainable = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tfb.Sigmoid().inverse(initial_guess['weights']))
            ),
            components_distribution=tfd.LogitNormal(
                loc=tf.Variable(tfb.Sigmoid().forward(initial_guess['means'].flatten())),
                scale=tfp_util.TransformedVariable(np.sqrt(initial_guess['covariances'].flatten()),
                                                   bijector=tfb.Softplus()),
            ),
            name=f'learnable_LogitNormal_obj'
        )
        return dist_trainable

    def fit(self, *,
            epochs: int = 5000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.01),
            **kwargs):
        fit_results = super().fit(epochs=epochs, optimiser=optimiser, **kwargs)

        constructor_params = {
            'mixture_distribution': {
                'name': 'Categorical',
                'params': {
                    'logits': fit_results.mixture_distribution.logits.numpy()
                }
            },
            'components_distribution': {
                'name': 'LogitNormal',
                'params': {
                    'loc': fit_results.components_distribution.loc.numpy(),
                    'scale': fit_results.components_distribution.scale.numpy()
                }
            }
        }
        save_pkl_file(self.fit_results_path, constructor_params)
        return fit_results


class MixtureVonMises(MixtureSameFamilyWrapper):
    """
    This is a scalar dist
    """

    def _get_dist_trainable(self, **kwargs):
        initial_guess = self._initial_guess_by_naive_gmm()
        dist_trainable = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tfb.Sigmoid().inverse(initial_guess['weights']))
            ),
            components_distribution=tfd.VonMises(
                loc=tf.Variable(initial_guess['means'].flatten()),

                concentration=tfp_util.TransformedVariable(1 / initial_guess['covariances'].flatten(),
                                                           bijector=tfb.Softplus())
            ),
            name=f'learnable_MixtureVonMises_obj'
        )
        return dist_trainable

    def fit(self, *,
            epochs: int = 5000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.01),
            **kwargs):
        fit_results = super().fit(epochs=epochs, optimiser=optimiser, **kwargs)

        constructor_params = {
            'mixture_distribution': {
                'name': 'Categorical',
                'params': {
                    'logits': fit_results.mixture_distribution.logits.numpy()
                }
            },
            'components_distribution': {
                'name': 'VonMises',
                'params': {
                    'loc': fit_results.components_distribution.loc.numpy(),
                    'concentration': fit_results.components_distribution.concentration.numpy()
                }
            }
        }
        save_pkl_file(self.fit_results_path, constructor_params)
        return fit_results


class MixtureWeibull(MixtureSameFamilyWrapper):
    """
    This is a scalar dist
    """

    def _get_dist_trainable(self, **kwargs):
        # initial_guess = self._initial_guess_by_naive_gmm()
        num_com = (6,)

        dist_trainable = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tf.random.uniform(num_com, 0.1, 0.2))
            ),
            components_distribution=tfd.Weibull(
                concentration=tfp_util.TransformedVariable(tf.random.uniform(num_com, 0.1, 10.2),
                                                           bijector=tfb.Softplus()),
                scale=tfp_util.TransformedVariable(tf.random.uniform(num_com, 0.1, 10.2),
                                                   bijector=tfb.Softplus())
            ),
            name=f'learnable_MixtureWeibull_obj'
        )
        return dist_trainable

    def fit(self, *,
            epochs: int = 25000,
            optimiser=tf.keras.optimizers.Adam(learning_rate=0.001),
            **kwargs):
        fit_results = super().fit(epochs=epochs, optimiser=optimiser, **kwargs)

        constructor_params = {
            'mixture_distribution': {
                'name': 'Categorical',
                'params': {
                    'logits': fit_results.mixture_distribution.logits.numpy()
                }
            },
            'components_distribution': {
                'name': 'Weibull',
                'params': {
                    'concentration': fit_results.components_distribution.concentration.numpy(),
                    'scale': fit_results.components_distribution.scale.numpy()
                }
            }
        }
        save_pkl_file(self.fit_results_path, constructor_params)
        return fit_results


class HighDimensionalReduce:
    __slots__ = ('ndarray_data', 'kmeans_file_')
    """
    默认ndarray_data的每一列代表一个维度，每一行代表一次记录
    """

    def __init__(self, ndarray_data: ndarray, kmeans_file_: Path = None):
        if ndarray_data.ndim == 1:
            ndarray_data = ndarray_data.reshape(-1, 1)
        self.ndarray_data = ndarray_data  # type: ndarray
        self.kmeans_file_ = kmeans_file_  # type: Path

    def dimension_reduction_by_kmeans(self, n_clusters: int = 5, **kmeans_args):
        @load_exist_pkl_file_otherwise_run_and_save(self.kmeans_file_)
        def fit_kmeans():
            nonlocal n_clusters
            n_clusters = kmeans_args.pop('n_clusters') if 'n_clusters' in kmeans_args else n_clusters
            kmeans = KMeans(n_clusters=n_clusters, **kmeans_args)
            return kmeans.fit(self.ndarray_data)

        kmeans_model = fit_kmeans  # type: KMeans

        return kmeans_model.predict(self.ndarray_data)

    def hierarchical_data_split(self, bin_step_in_tuple: Tuple[float, ...]) -> Tuple[int, ...]:
        pass


if __name__ == "__main__":
    # %% Test for MixtureLogitNormal
    """
    mix_mvntril = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=[0.3, 0.35, 0.34]),
        components_distribution=tfd.LogitNormal(
            loc=[0, 0.5, 1],
            scale=[6.2, 2.5, 3.15]),
        name='learnable_mix')
    samples = np.clip(mix_mvntril.sample(100000).numpy(), 1e-6, 1 - 1e-6)
    hist(samples, bins=np.arange(0., 1, 0.01), title='true')
    dist_f = MixtureLogitNormal(data=samples[..., np.newaxis], fit_results_path=Path(r'.\my_fit.pkl'),
                                quantile_meta={'look_up_tbl_path': Path(r'.\my_fit_look_up_tbl.pkl'),
                                               'cdf_x_min': 0,
                                               'cdf_x_max': 1,
                                               'cdf_x_num': 1_000_000}
                                )
    dist_f.fit(epochs=1000)
    dist_f.load()
    hist(dist_f.sample(100000).numpy(), bins=np.arange(0., 1, 0.01), title='samples')
    series(np.arange(-3, 25, 0.01), dist_f.cdf(np.arange(-3, 25, 0.01)).numpy(), title='fit cdf')
    series(np.arange(0, 100, 0.1), dist_f.quantile(np.arange(0, 100, 0.1)), title='fit quantile')
    """

    # %% Test for MixtureWeibull
    mix_mvntril = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=[0.3, 3, 0.4, 0.2, 0.3, 0.9]),
        components_distribution=tfd.Weibull(
            concentration=[3, 0.5, 1, 2, 1.5, 2.6],
            scale=[6.2, 2.5, 3.15, 3.7, 9.8, 6.9]),
        name='learnable_mix')
    samples = mix_mvntril.sample(100000).numpy()
    hist(samples, bins=np.arange(0., 10, 0.1), title='true')
    dist_f = MixtureWeibull(data=samples[..., np.newaxis], fit_results_path=Path(r'.\my_fit.pkl'),
                                    quantile_meta={'look_up_tbl_path': Path(r'.\my_fit_look_up_tbl.pkl'),
                                                   'cdf_x_min': 0,
                                                   'cdf_x_max': 10.,
                                                   'cdf_x_num': 1_000_000}
                                    )
    dist_f.fit()
    dist_f.load()
    hist(dist_f.sample(100000).numpy(), bins=np.arange(0., 10, 0.1), title='samples')
    series(np.arange(-3, 25, 0.01), dist_f.cdf(np.arange(-3, 25, 0.01)).numpy(), title='fit cdf')
    series(np.arange(0, 100, 0.1), dist_f.quantile(np.arange(0, 100, 0.1)), title='fit quantile')
