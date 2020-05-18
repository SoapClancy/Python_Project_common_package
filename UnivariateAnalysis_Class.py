import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from numpy import ndarray
from typing import Tuple, Union, Iterable
from Ploting.fast_plot_Func import hist, series
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from Data_Preprocessing.float_precision_control_Func import get_decimal_places_of_float
from abc import ABCMeta, abstractmethod
from scipy.interpolate import interp1d
from collections import OrderedDict
import re


class Univariate:
    __slots__ = ('univariate_data', 'univariate_name')

    def __init__(self, univariate_data: ndarray = None, univariate_name: str = None, **kwargs):
        if (not isinstance(univariate_data, ndarray)) and (univariate_data is not None):
            raise Exception(
                "'univariate_data' should be ndarray instead of {}".format(univariate_data.__class__.__name))
        if (isinstance(univariate_data, ndarray)) and (univariate_data.ndim != 1):
            raise Exception("'Univariate' class should be only 1-dimensional (maybe try to flatten at first!)")
        self.univariate_data = univariate_data
        self.univariate_name = univariate_name  # type: str

    def __str__(self):
        return "{} ndarray with size = {}".format(self.univariate_name or 'Unnamed', self.univariate_data.size)

    def plot_hist(self, *args, **kwargs):
        return hist(self.univariate_data, *args, **kwargs)

    def plot_ecdf(self, x: ndarray = None, **kwargs):
        return series(x, self.ecdf(x), **kwargs)

    def ecdf(self, x: ndarray = None):
        ecdf = ECDF(self.univariate_data)
        x = x if x is not None else np.arange(self.univariate_data.min(), self.univariate_data.max(),
                                              0.1 ** get_decimal_places_of_float(self.univariate_data[0]) + 1)
        return ecdf(x)

    def cal_inverse_ecdf_as_look_up_table(self, accuracy: float = 0.01) -> ndarray:
        inverse_ecdf = np.arange(0, 1 + accuracy, accuracy).reshape(-1, 1)
        inverse_ecdf = np.concatenate((inverse_ecdf, np.full_like(inverse_ecdf, np.nan)), axis=1)
        for i in range(inverse_ecdf.shape[0]):
            inverse_ecdf[i, 1] = np.nanpercentile(self.univariate_data, inverse_ecdf[i, 0] * 100)
        return inverse_ecdf

    def find_nearest_inverse_ecdf(self, ecdf_value: Union[float, ndarray]) -> ndarray:
        def find_nearest_inverse_ecdf_one(ecdf_value_one):
            idx = np.argmin(abs(self.cal_inverse_ecdf_as_look_up_table()[:, 0] - ecdf_value_one))
            return self.cal_inverse_ecdf_as_look_up_table()[idx, 1]

        if isinstance(ecdf_value, float):
            return np.array([find_nearest_inverse_ecdf_one(ecdf_value)])
        else:
            return np.array([find_nearest_inverse_ecdf_one(x) for x in ecdf_value])

    def fit_using_gaussian_mixture_model(self, max_mixture_number: int = 15, **gmm_args):
        if self.univariate_data.size <= 3:
            return None
        fitting_data = self.univariate_data.reshape(-1, 1)
        bic = []
        lowest_bic = np.inf
        best_gmm = None
        max_iter = gmm_args.pop('max_iter') if 'max_iter' in gmm_args else 512  # 设置max_iter的默认值并修改gmm_args
        for i in range(max_mixture_number):
            gmm = GaussianMixture(n_components=i + 1, covariance_type='full',
                                  max_iter=max_iter, **gmm_args)
            if fitting_data.size >= (i + 1):
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        gmm.fit(fitting_data)
                        bic.append(gmm.bic(fitting_data))
                    except Warning:
                        bic.append(np.inf)
            else:
                bic.append(np.inf)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
        return best_gmm  # type:GaussianMixture

    def fit_using_bayesian_gaussian_mixture_model(self, max_mixture_number: int = 15, **gmm_args):
        fitting_data = self.univariate_data.reshape(-1, 1)
        if fitting_data.size <= 3:
            return None
        max_iter = gmm_args.pop('max_iter') if 'max_iter' in gmm_args else 256
        bgmm = BayesianGaussianMixture(n_components=max_mixture_number, covariance_type='full',
                                       max_iter=max_iter, **gmm_args).fit(fitting_data)
        return bgmm


class CategoryUnivariate(Univariate):
    def __init__(self, univariate_data: ndarray, univariate_name: str = None):
        if np.any(np.isnan(univariate_data)):
            warnings.warn('np.nan is found in the categorical data', UserWarning)
        if not univariate_data.dtype == np.dtype('int'):
            warnings.warn("The categorical data is not type 'int'", UserWarning)
        super().__init__(univariate_data, univariate_name)

    def __str__(self):
        return "{} category ndarray with {} categories and size = {}".format(self.univariate_name or 'Unnamed',
                                                                             self.category.size,
                                                                             self.univariate_data.size)

    @property
    def category(self):
        return np.unique(self.univariate_data)

    def update_category(self, old_category: int, new_category: int):
        """
        修改索引来的值相当于直接修改内存
        """
        if new_category in self.category:
            raise Exception("'new_category' is already existing")
        else:
            idx = self.univariate_data == old_category
            self.univariate_data[idx] = new_category
        return self.univariate_data

    def cal_tuple_category_mask(self, tuple_category: Tuple[int, ...]):
        mask = np.full(self.univariate_data.shape, False)
        for this_category in tuple_category:
            mask = np.bitwise_or(mask, self.univariate_data == this_category)
        return mask


class UnivariateProbabilisticModel(Univariate, metaclass=ABCMeta):
    __slots__ = ('theoretic_min_value', 'theoretic_max_value')

    def __init__(self, *, theoretic_min_value: float = np.NINF, theoretic_max_value: float = np.inf, **kwargs):
        super().__init__(**kwargs)
        self.theoretic_min_value = theoretic_min_value  # type: float
        self.theoretic_max_value = theoretic_max_value  # type: float

    @abstractmethod
    def sample(self, number_of_samples: int) -> ndarray:
        pass

    @abstractmethod
    def pdf_estimate(self, x: ndarray, **kwargs) -> ndarray:
        pass

    @abstractmethod
    def cdf_estimate(self, x: ndarray) -> ndarray:
        pass

    def cdf_estimate_by_sampling_method(self, *, x: ndarray, number_of_samples: int = 10_000_000):
        pass
        # """
        # 测试阶段(stupid copula)才需要设置np.random.seed(1)
        # """
        # np.random.seed(1)
        sample_results = self.sample(number_of_samples=number_of_samples)
        ecdf = ECDF(sample_results)
        cdf_estimate = ecdf(x)
        cdf_estimate[np.isnan(x)] = np.nan  # ecdf函数对于np.nan的输出是1，也不知道是什么鬼操作...，所以手动调一下
        return cdf_estimate

    @property
    @abstractmethod
    def mean_(self):
        pass

    @abstractmethod
    def cal_inverse_cdf_as_look_up_table(self, accuracy: float = 0.01) -> ndarray:
        pass

    @abstractmethod
    def find_nearest_inverse_cdf(self, cdf_value: Union[float, ndarray]) -> ndarray:
        pass

    def cal_uncertainty_interval_width(self):
        result = OrderedDict().fromkeys(('0_100', '5_95', '10_90', '15_85',
                                         '20_80', '25_75', '30_70', '35_65',
                                         '40_60', '45_55'))
        for key in result.keys():
            if key == '0_100':
                result[key] = self.find_nearest_inverse_cdf(1.0 - 0.0001) - self.find_nearest_inverse_cdf(0.0001)
            else:
                lower_cdf = float(re.search(r'^.*_', key).group(0)[:-1]) / 100
                upper_cdf = float(re.search(r'_.*$', key).group(0)[1:]) / 100
                result[key] = self.find_nearest_inverse_cdf(upper_cdf) - self.find_nearest_inverse_cdf(lower_cdf)
        return result

    def cal_inverse_cdf_as_look_up_table_by_sampling_method(self, number_of_samples: int = 10_000_000):
        return Univariate(self.sample(number_of_samples=number_of_samples)).cal_inverse_ecdf_as_look_up_table()

    def find_nearest_inverse_ecdf_by_sampling_method(self, cdf_value: Union[float, ndarray], *,
                                                     number_of_samples: int = 50_000) -> ndarray:
        np.random.seed(1)
        univariate = Univariate(self.sample(number_of_samples=number_of_samples))
        return univariate.find_nearest_inverse_ecdf(cdf_value)

    def plot_pdf(self, x: ndarray, pdf_value=None, show_hist: bool = False, ax=None, *args, **kwargs):
        pdf_value = pdf_value if pdf_value is not None else self.pdf_estimate(x=x)
        if show_hist:
            if not isinstance(self.univariate_data, ndarray):
                raise Exception("Should specify 'univariate_data' to include the histogram plot")
            else:
                protect = kwargs.pop('label') if 'label' in kwargs else None
                bins = kwargs.pop('bins') if 'bins' in kwargs else None
                ax = self.plot_hist(ax, density=True, color='darkorange', label='histogram', bins=bins, *args, **kwargs)
                kwargs.setdefault('label', protect)
        color = kwargs.pop('color') if 'color' in kwargs else 'k'
        linestyle = kwargs.pop('linestyle') if 'linestyle' in kwargs else '-'
        return series(x, pdf_value, ax=ax, y_label='Probability density', color=color, linestyle=linestyle,
                      linewidth=1, *args, **kwargs)

    def plot_cdf(self, x: ndarray, *, show_ecdf: bool = False, **kwargs):
        cdf_value = self.cdf_estimate(x=x)
        if show_ecdf:
            if not isinstance(self.univariate_data, ndarray):
                raise Exception("Should specify 'univariate_data' to include the empirical cdf plot")
            else:
                protect = kwargs.pop('label') if 'label' in kwargs else None
                ax = self.plot_ecdf(x=x, color='r', linestyle='--', linewidth=1.0, label='Empirical cdf', **kwargs)
                kwargs.setdefault('label', protect)
        else:
            ax = None
        return series(x, cdf_value, ax=ax, y_label='Cumulative probability', color='k', linestyle='-',
                      linewidth=0.5, **kwargs)


class UnivariatePDFOrCDFLike(UnivariateProbabilisticModel):
    __slots__ = ('pdf_like_ndarray', 'pmf_like_ndarray', 'cdf_like_ndarray')

    def __init__(self, *, pdf_like_ndarray: ndarray = None, cdf_like_ndarray: ndarray = None,
                 super_sampling_and_renormalise: bool = True,
                 theoretic_min_value: float = np.NINF, theoretic_max_value: float = np.inf, **kwargs):
        super().__init__(theoretic_min_value=theoretic_min_value, theoretic_max_value=theoretic_max_value, **kwargs)
        """
        早晚要用元编程改写__new__
        """
        # 已知pdf的__init__，先super_sampling_and_renormalise再求cdf
        if pdf_like_ndarray is not None:
            if super_sampling_and_renormalise:
                self.pdf_like_ndarray = self.__super_sampling(pdf_like_ndarray)
                self.renormalise()
            else:
                self.pdf_like_ndarray = pdf_like_ndarray
            bin_width = self.__cal_bin_width(self.pdf_like_ndarray)
            bin_height = self.__cal_bin_height_in_terms_of_pdf(self.pdf_like_ndarray)
            self.pmf_like_ndarray = np.stack((bin_width * bin_height, self.pdf_like_ndarray[1:, 1]), axis=1)
            self.cdf_like_ndarray = np.stack((np.cumsum(self.pmf_like_ndarray[:, 0]),
                                              self.pdf_like_ndarray[1:, 1]), axis=1)
        if cdf_like_ndarray is not None:
            self.cdf_like_ndarray = cdf_like_ndarray
            pmf = self.__cal_bin_height_in_terms_of_cdf(self.cdf_like_ndarray)
            self.pmf_like_ndarray = np.stack((pmf, self.cdf_like_ndarray[1:, 1]), axis=1)

    @property
    def mean_(self):
        return float(np.nansum(self.pmf_like_ndarray[:, 0] * self.pmf_like_ndarray[:, 1]))

    def sample(self, number_of_samples: int) -> ndarray:
        pass

    def pdf_estimate(self, x: ndarray, **kwargs) -> ndarray:
        pass

    def cdf_estimate(self, x: ndarray) -> ndarray:
        pass

    def cal_inverse_cdf_as_look_up_table(self, accuracy: float = 0.01) -> ndarray:
        pass

    def find_nearest_inverse_cdf(self, cdf_value: Union[float, ndarray]) -> ndarray:
        inverse_cdf = []
        if isinstance(cdf_value, float):
            cdf_value = np.array([cdf_value])
        for i in cdf_value:
            inverse_cdf.append(self.cdf_like_ndarray[np.argmin(np.abs(self.cdf_like_ndarray[:, 0] - i)), 1])
        return np.array(inverse_cdf)

    @staticmethod
    def __cal_bin_width(pdf_or_cdf_like_ndarray: ndarray):
        return pdf_or_cdf_like_ndarray[1:, 1] - pdf_or_cdf_like_ndarray[:-1, 1]

    @staticmethod
    def __cal_bin_height_in_terms_of_pdf(pdf_like_ndarray: ndarray):
        return np.nanmean(np.concatenate((pdf_like_ndarray[1:, 0].reshape(-1, 1),
                                          pdf_like_ndarray[:-1, 0].reshape(-1, 1)), axis=1), axis=1)

    @staticmethod
    def __cal_bin_height_in_terms_of_cdf(cdf_like_ndarray: ndarray):
        return cdf_like_ndarray[1:, 0] - cdf_like_ndarray[:-1, 0]

    def renormalise(self):
        if np.any(self.pdf_like_ndarray[:, 0] < 0):
            self.pdf_like_ndarray[:, 0][self.pdf_like_ndarray[:, 0] < 0] = 10e-12
            # raise Exception("Probability density cannot be smaller than 0")
        bin_width = self.__cal_bin_width(self.pdf_like_ndarray)
        bin_height = self.__cal_bin_height_in_terms_of_pdf(self.pdf_like_ndarray)
        sum_under_curve = np.nansum(bin_width * bin_height)
        self.pdf_like_ndarray[:, 0] = self.pdf_like_ndarray[:, 0] / sum_under_curve
        return self.pdf_like_ndarray

    @staticmethod
    def __super_sampling(pdf_like_ndarray, linspace_number: int = 5000):
        # 提升self.pdf_like_ndarray的精度
        pdf_like_ndarray_hi = np.full((int(linspace_number), 2), np.nan)
        pdf_like_ndarray_hi[:, 1] = np.linspace(np.nanmin(pdf_like_ndarray[:, 1]),
                                                np.nanmax(pdf_like_ndarray[:, 1]), linspace_number)
        pdf_like_ndarray_hi[:, 0] = interp1d(pdf_like_ndarray[:, 1], pdf_like_ndarray[:, 0])(
            pdf_like_ndarray_hi[:, 1])
        return pdf_like_ndarray_hi

    def plot_cdf_like_ndarray(self, *args, **kwargs):
        return series(self.cdf_like_ndarray[:, 1], self.cdf_like_ndarray[:, 0], *args, **kwargs)

    def plot_pmf_like_ndarray(self, *args, **kwargs):
        return series(self.pmf_like_ndarray[:, 1], self.pmf_like_ndarray[:, 0], *args, **kwargs)

    def plot_pdf_like_ndarray(self, *args, **kwargs):
        return series(self.pdf_like_ndarray[:, 1], self.pdf_like_ndarray[:, 0], *args, **kwargs)


class MixtureUnivariatePDFOrCDFLike:
    @classmethod
    def do_mixture(cls, univariate_pdf_or_cdf_like: ndarray,
                   weights: tuple = None, target: bool = 'cdf'):
        if target == 'cdf':
            xx = univariate_pdf_or_cdf_like[0].cdf_like_ndarray[:, 1]
            mixture_cdf_like_ndarray_yy = np.full(
                univariate_pdf_or_cdf_like[0].cdf_like_ndarray.shape[0], 0.
            )
            for i in range(univariate_pdf_or_cdf_like.shape[0]):
                if weights is not None:
                    mixture_cdf_like_ndarray_yy += weights[i] * univariate_pdf_or_cdf_like[i].cdf_like_ndarray[:, 0]
                else:
                    equal_weight = 1. / univariate_pdf_or_cdf_like.shape[0]
                    mixture_cdf_like_ndarray_yy += equal_weight * univariate_pdf_or_cdf_like[i].cdf_like_ndarray[:, 0]
            return UnivariatePDFOrCDFLike(cdf_like_ndarray=np.stack((
                mixture_cdf_like_ndarray_yy, xx), axis=1))


class DeterministicUnivariateProbabilisticModel(UnivariateProbabilisticModel):
    delta_around_value = 10e-8
    __slots__ = ('value',)

    def __init__(self, value: float, **kwargs):
        super().__init__(theoretic_min_value=value - self.delta_around_value,
                         theoretic_max_value=value + self.delta_around_value, **kwargs)
        self.value = value

    def sample(self, number_of_samples: int) -> ndarray:
        return np.random.uniform(low=self.theoretic_min_value,
                                 high=self.theoretic_max_value, size=number_of_samples)

    def pdf_estimate(self, x: ndarray, **kwargs) -> ndarray:
        pdf_estimate_results = np.full(x.shape, np.nan)
        mask_out_range = np.bitwise_or(x >= self.theoretic_max_value,
                                       x < self.theoretic_min_value)
        mask_in_range = ~mask_out_range
        pdf_estimate_results[mask_out_range] = 0
        pdf_estimate_results[mask_in_range] = 1 / (self.theoretic_max_value - self.theoretic_min_value)
        return pdf_estimate_results

    def cdf_estimate(self, x: ndarray) -> ndarray:
        cdf_estimate_results = np.full(x.shape, np.nan)
        mask_lower_range = x < self.theoretic_min_value
        mask_higher_range = x >= self.theoretic_max_value
        mask_in_range = np.bitwise_and(x >= self.theoretic_min_value,
                                       x < self.theoretic_max_value)
        cdf_estimate_results[mask_higher_range] = 1
        cdf_estimate_results[mask_lower_range] = 0
        cdf_estimate_results[mask_in_range] = interp1d(
            np.array([self.theoretic_min_value, self.theoretic_max_value]),
            np.array([0, 1]))(x[mask_in_range]).flatten()
        return cdf_estimate_results

    @property
    def mean_(self):
        return self.value

    def cal_inverse_cdf_as_look_up_table(self, accuracy: float = 0.01) -> ndarray:
        pass

    def find_nearest_inverse_cdf(self, cdf_value: Union[float, ndarray]) -> ndarray:
        if isinstance(cdf_value, float):
            cdf_value = np.array([cdf_value])
        inverse_cdf = interp1d(np.array([0, 1]),
                               np.array([self.theoretic_min_value, self.theoretic_max_value]))(cdf_value).flatten()
        return inverse_cdf


class UnivariateGaussianMixtureModel(UnivariateProbabilisticModel):
    __slots__ = ('gmm',)

    def __init__(self, gmm: GaussianMixture = None, **kwargs):
        super().__init__(**kwargs)
        self.gmm = gmm  # type:GaussianMixture

    def __str__(self):
        return 'GMM with {} component(s), and weights = {}'.format(self.gmm.n_components,
                                                                   self.gmm.weights_)

    @property
    def mean_(self) -> float:
        mean_values = np.array([x for x in self.gmm.means_])
        weight_values = np.array([x for x in self.gmm.weights_])
        return float(np.sum(mean_values.flatten() * weight_values.flatten()))

    def sample(self, number_of_samples: int) -> ndarray:
        def gmm_sampler(x: GaussianMixture):
            new_samples = x.sample(out_of_bound_idx_num.size)[0].flatten()
            return new_samples

        try:
            sample_results = self.gmm.sample(number_of_samples)[0].flatten()
        except ValueError:
            try:
                warnings.warn('Sampling ValueError, turn to empirical solution', UserWarning)
                sample_results = np.random.choice(self.univariate_data, number_of_samples)
            except ValueError:
                warnings.warn("Sampling ValueError, and no 'univariate_data' is assigned, output the mean value of gmm",
                              UserWarning)
                sample_results = np.full(number_of_samples, self.mean_)

        out_of_bound_idx_num = np.argwhere(np.bitwise_or(sample_results < self.theoretic_min_value,
                                                         sample_results > self.theoretic_max_value)).flatten()
        while out_of_bound_idx_num.size > 0:
            sample_results[out_of_bound_idx_num] = gmm_sampler(self.gmm)
            out_of_bound_idx_num = np.argwhere(np.bitwise_or(sample_results < self.theoretic_min_value,
                                                             sample_results > self.theoretic_max_value)).flatten()

        return sample_results

    def pdf_estimate(self, x: ndarray, **kwargs):
        logprob = self.gmm.score_samples(x.reshape(-1, 1))
        pdf_value = np.exp(logprob)
        return pdf_value

    def cdf_estimate(self, x: ndarray):
        return self.cdf_estimate_by_sampling_method(x=x, number_of_samples=10_000_000)

    def cal_inverse_cdf_as_look_up_table(self, accuracy: float = 0.01) -> ndarray:
        return self.cal_inverse_cdf_as_look_up_table_by_sampling_method()

    def find_nearest_inverse_cdf(self, cdf_value: Union[float, ndarray],
                                 number_of_samples: int = 50_000) -> ndarray:
        return self.find_nearest_inverse_ecdf_by_sampling_method(cdf_value, number_of_samples=number_of_samples)


class UnivariateMixtureOfGaussianMixtureModel(UnivariateGaussianMixtureModel, UnivariateProbabilisticModel):
    __slots__ = ('tuple_of_hyper_gmms', 'tuple_of_hyper_weights')

    def __init__(self, tuple_of_hyper_gmms: Tuple[GaussianMixture, ...], tuple_of_hyper_weights: Tuple[float, ...],
                 **kwargs):
        # 先初始化Univariate类，注意python多重继承的MRO！
        super().__init__(**kwargs)
        self.tuple_of_hyper_gmms = tuple_of_hyper_gmms  # Tuple[GaussianMixture, ...]
        self.tuple_of_hyper_weights = tuple_of_hyper_weights  # Tuple[float, ...]

    def __str__(self):
        return str(self.tuple_of_hyper_weights.__len__()) + ' hyper GMMs with hyper weights = ' + \
               str(self.tuple_of_hyper_weights)

    @property
    def hyper_size(self):
        return self.tuple_of_hyper_weights.__len__()

    def sample(self, *, number_of_samples: int) -> ndarray:
        """
        采样函数。应用了两种采样方法，应对不同的样本数
        """
        sample_results = []
        number_of_samples_per_model = np.array([int(number_of_samples * this_weight)
                                                for this_weight in self.tuple_of_hyper_weights])
        for i, this_model in enumerate(self.tuple_of_hyper_gmms):
            temp_model = UnivariateGaussianMixtureModel(this_model,
                                                        theoretic_min_value=self.theoretic_min_value,
                                                        theoretic_max_value=self.theoretic_max_value)
            temp_sample = temp_model.sample(number_of_samples=number_of_samples_per_model[i])
            sample_results.append(temp_sample)
        sample_results = np.array(sample_results).flatten()
        return sample_results

    def pdf_estimate(self, x: ndarray, *, cal_components: bool = False, **kwargs):
        logprob = np.array([this_model.score_samples(x.reshape(-1, 1)) for this_model in self.tuple_of_hyper_gmms])
        individual_pdf = np.exp(logprob)
        model_weights = np.array(self.tuple_of_hyper_weights).reshape(-1, 1)
        weighted_individual_pdf = individual_pdf * model_weights
        aggregate_pdf = np.sum(weighted_individual_pdf, axis=0)
        if cal_components:
            return aggregate_pdf, weighted_individual_pdf, individual_pdf
        return aggregate_pdf
