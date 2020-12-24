from Filtering.simple_filtering_Func import interquartile_outlier, shut_down_outlier
from Data_Preprocessing.float_precision_control_Func import get_decimal_places_of_float, \
    convert_ndarray_to_arbitrary_precision
from Ploting.fast_plot_Func import *
from Ploting.classification_plot_Func import bivariate_classification_scatter
from Ploting.uncertainty_plot_Func import bivariate_uncertainty_plot
import copy
from typing import Tuple, Callable, Union, Iterable
from Filtering.sklearn_novelty_and_outlier_detection_Func import use_isolation_forest, use_local_outlier_factor, \
    use_optics_maximum_size
from UnivariateAnalysis_Class import Univariate, UnivariateGaussianMixtureModel
from sklearn.mixture import GaussianMixture
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike
from ConvenientDataType import UncertaintyDataFrame, StrOneDimensionNdarray
import warnings
from itertools import chain
import pandas as pd
from Data_Preprocessing.float_precision_control_Func import covert_to_str_one_dimensional_ndarray


class MethodOfBins:
    __slots__ = ('predictor_var', 'dependent_var', 'bin_step', 'array_of_bin_boundary',
                 '__mob', 'considered_data_mask_for_mob_calculation')

    def __init__(self, predictor_var: ndarray, dependent_var: ndarray,
                 *, bin_step: float, first_bin_left_boundary: float = None, last_bin_left_boundary: float = None,
                 considered_data_mask_for_mob_calculation: ndarray = None, **kwargs):
        """
        :param considered_data_mask_for_mob_calculation: 有时候可能只需要考虑一部分数据。尤其在分层outlier identification中。
        """
        self.predictor_var = predictor_var  # type: ndarray
        self.dependent_var = dependent_var  # type: ndarray
        self.bin_step = bin_step
        if bin_step is not None:
            if first_bin_left_boundary is None:
                first_bin_left_boundary = np.nanmin(self.predictor_var) - bin_step / 2
            if last_bin_left_boundary is None:
                last_bin_left_boundary = np.nanmax(self.predictor_var) - bin_step / 2
            self.array_of_bin_boundary = self.cal_array_of_bin_boundary(first_bin_left_boundary,
                                                                        last_bin_left_boundary,
                                                                        self.bin_step)
            self.considered_data_mask_for_mob_calculation = considered_data_mask_for_mob_calculation
            self.__mob = self.__cal_mob()

    def __str__(self):
        return "MethodOfBins instance for {} recordings".format(self.predictor_var.size)

    @staticmethod
    def cal_array_of_bin_boundary(first_bin_left_boundary: float,
                                  last_bin_left_boundary: float,
                                  bin_step: float):
        """
        计算mob的每个bin的左右边界值和中间值
        """

        bin_left_boundary = np.arange(first_bin_left_boundary, last_bin_left_boundary + bin_step, bin_step)
        bin_medium_boundary = bin_left_boundary + bin_step / 2
        bin_right_boundary = bin_left_boundary + bin_step
        results = np.array([bin_left_boundary, bin_medium_boundary, bin_right_boundary]).T
        results = convert_ndarray_to_arbitrary_precision(results, get_decimal_places_of_float(bin_step) + 1)
        return results

    def __cal_mob(self) -> dict:
        """
        计算mob。不应该被外部调用。外部如果想访问mob的话应该读取mob属性
        """
        predictor_var, dependent_var = copy.deepcopy(self.predictor_var), copy.deepcopy(self.dependent_var)
        if self.considered_data_mask_for_mob_calculation is not None:
            predictor_var = predictor_var.astype(float)
            dependent_var = dependent_var.astype(float)
            predictor_var[~self.considered_data_mask_for_mob_calculation] = np.nan
            dependent_var[~self.considered_data_mask_for_mob_calculation] = np.nan
        total_valid_data_number = 0
        dict_index, inner_dict = [], []
        for this_bin_idx, this_bin_boundary in enumerate(self.array_of_bin_boundary):
            dict_index.append(this_bin_idx)
            bin_data_idx = np.bitwise_and(predictor_var >= self.array_of_bin_boundary[this_bin_idx, 0],
                                          predictor_var < self.array_of_bin_boundary[this_bin_idx, -1])
            bin_data = dependent_var[bin_data_idx]
            bin_frequency = bin_data.size
            total_valid_data_number += bin_frequency
            inner_dict.append({'this_bin_boundary': this_bin_boundary,
                               'this_bin_is_empty': False if bin_frequency > 0 else True,
                               'this_bin_frequency': bin_frequency,
                               'this_bin_var_idx': np.where(bin_data_idx)[0],
                               'dependent_var_in_this_bin': bin_data})
        mob = dict(zip(dict_index, inner_dict))
        # 计算每个bin的probability
        for key, val in mob.items():
            mob[key]['this_bin_probability'] = val['this_bin_frequency'] / total_valid_data_number
        return mob

    @property
    def mob(self):
        return self.__mob

    def identify_interquartile_outliers(self):
        """
        看每个bin的outlier的情况，并反映射到原始数据对的索引。True表示outlier
        :return: 原始数据对的outlier的索引
        """
        outlier = np.full(self.predictor_var.shape, False)
        for this_bin in self.mob.values():
            if this_bin['this_bin_frequency'] == 0:
                continue
            outlier_bool_idx_in_this_bin = interquartile_outlier(this_bin['dependent_var_in_this_bin'])
            outlier_idx = this_bin['this_bin_var_idx'][outlier_bool_idx_in_this_bin]
            outlier[outlier_idx] = True
        return outlier

    def __fit_mob(self, func: str, **kwargs) -> dict:
        """
        用单变量分布去拟合每一个bin。可支持多种方法。作为内部调用的函数
        """
        mob_fitting = {}
        for key, this_bin in self.mob.items():
            mob_fitting.setdefault(key, {'this_bin_boundary': this_bin['this_bin_boundary'],
                                         'this_bin_is_empty': True,
                                         'this_bin_probability_model': None})
            if this_bin['this_bin_is_empty']:
                continue
            else:
                univariate = Univariate(this_bin['dependent_var_in_this_bin'])
                if func == 'using_gaussian_mixture_model':
                    model = univariate.fit_using_gaussian_mixture_model(**kwargs)
                    mob_fitting[key]['this_bin_probability_model'] = model
                    if model is not None:
                        mob_fitting[key]['this_bin_is_empty'] = False
        return mob_fitting

    def fit_mob_using_gaussian_mixture_model(self, **gmm_args) -> dict:
        return self.__fit_mob('using_gaussian_mixture_model', **gmm_args)

    @staticmethod
    def find_mob_key_according_to_mob_or_mob_fitting_like_dict(predictor_var_value: float,
                                                               mob_or_mob_fitting_like: dict) -> dict:
        """
        找出离给定predictor_var_value最近的一系列bin。None值的bin不会被考虑
        :param predictor_var_value: 带判定的属于哪个bin的数值
        :param mob_or_mob_fitting_like: key是int索引，然后value是None或者dict，
        而且子dict中必须包含‘this_bin_boundary’字段
        :return: 可能的bin的索引：
        'accurate_bin_key'表示不考虑有没有有效模型，predictor_var_value对应的bin的key
        'nearest_not_none_bin_keys'表示最近的有有效有效模型的bin的key
        'not_none_bin_keys'表示所有的有有效有效模型的bin的key，距离从近到远
        """
        possible_bin_keys = []
        for key, value in mob_or_mob_fitting_like.items():
            possible_bin_keys.append({'this_bin_key': key,
                                      'this_bin_distance': abs(value['this_bin_boundary'][1] - predictor_var_value),
                                      'this_bin_is_empty': value['this_bin_is_empty']})
        possible_bin_keys.sort(key=lambda x: x['this_bin_distance'], reverse=False)
        accurate_bin_key = possible_bin_keys[0]['this_bin_key']
        not_none_bin_keys = [x['this_bin_key'] for x in possible_bin_keys if not x['this_bin_is_empty']]
        return {'accurate_bin_key': accurate_bin_key,
                'nearest_not_none_bin_keys': not_none_bin_keys[0],
                'not_none_bin_keys': tuple(not_none_bin_keys)}

    def cal_mob_statistic_eg_quantile(self, statistic='mean', behaviour='deprecated') -> Union[ndarray,
                                                                                               UncertaintyDataFrame]:
        """
        Calculate the statistics in each bin
        :param statistic: Can either be str or Iterable obj (to calculate the quantiles, e.g., 0.5 represents median)
        :param behaviour: This parameter has no effect, is deprecated, and will be removed.
        It's here for back-compatibility purpose.
        'deprecated' is used by default and for previous codes compatibility.
        'new' will cause the function to return a UncertaintyDataFrame obj
        :return:
        """
        assert (behaviour in ('deprecated', 'new')), "behaviour must be in ('deprecated', 'new')"
        if behaviour == 'deprecated':
            # Deprecated
            warnings.warn("Deprecated", DeprecationWarning)
            if isinstance(statistic, str) and (statistic == 'mean'):
                mob_statistic = np.full((self.array_of_bin_boundary.shape[0], 2), np.nan)
                mob_statistic[:, 0] = self.array_of_bin_boundary[:, 1]
                for i, this_bin in enumerate(self.mob.values()):
                    if this_bin['this_bin_is_empty']:
                        continue
                    else:
                        mob_statistic[i, 1] = np.nanmean(this_bin['dependent_var_in_this_bin'])
            elif isinstance(statistic, Iterable):
                mob_statistic = np.full((self.array_of_bin_boundary.shape[0], len(statistic) + 1), np.nan)
                mob_statistic[:, 0] = self.array_of_bin_boundary[:, 1]
                for i, this_bin in enumerate(self.mob.values()):
                    if this_bin['this_bin_is_empty']:
                        continue
                    else:
                        mob_statistic[i, 1:] = np.nanpercentile(this_bin['dependent_var_in_this_bin'],
                                                                np.array(statistic) * 100)
            else:
                raise Exception('Unsupported statistic')
        else:
            # New. Instead of deprecated
            mob_statistic = UncertaintyDataFrame(index=list(
                chain(covert_to_str_one_dimensional_ndarray(
                    np.array(statistic) * 100, '0.001') if statistic is not None else [],
                      ['mean', 'std.'])
            ),
                columns=self.array_of_bin_boundary[:, 1])
            for i, this_bin in enumerate(self.mob.values()):
                if this_bin['this_bin_is_empty']:
                    continue
                else:
                    mob_statistic.iloc[0:-2, i] = np.nanpercentile(
                        this_bin['dependent_var_in_this_bin'], np.array(mob_statistic.index[0:-2]).astype('float')
                    )
                    mob_statistic.loc['mean', this_bin['this_bin_boundary'][1]] = np.nanmean(
                        this_bin['dependent_var_in_this_bin']
                    )
                    mob_statistic.loc['std.', this_bin['this_bin_boundary'][1]] = np.nanstd(
                        this_bin['dependent_var_in_this_bin']
                    )

        return mob_statistic

    def plot_mob_statistic(self, show_scatter: bool = True, statistic='mean', ax=None, **kwargs):
        mob_statistic = self.cal_mob_statistic_eg_quantile(statistic)
        scatter_color = kwargs.pop('scatter_color') if 'scatter_color' in kwargs else 'g'
        scatter_ax = scatter(self.predictor_var[self.considered_data_mask_for_mob_calculation],
                             self.dependent_var[self.considered_data_mask_for_mob_calculation],
                             ax,
                             marker='.', c=scatter_color, alpha=0.5,
                             s=4) if show_scatter else None
        series_color = kwargs.pop('series_color') if 'series_color' in kwargs else 'r'
        series_linestyle = kwargs.pop('series_linestyle') if 'series_linestyle' in kwargs else '-'
        return series(mob_statistic[:, 0], mob_statistic[0:, 1], scatter_ax,
                      color=series_color, linestyle=series_linestyle, **kwargs)

    def plot_mob_uncertainty(self, show_scatter: bool = True, **kwargs):
        scatter_ax = scatter(self.predictor_var, self.dependent_var, marker='.', c='b', alpha=0.1,
                             s=4) if show_scatter else None
        boundary = np.arange(0, 1.05, 0.05)
        x, y = [], []
        for this_bin in self.mob.values():
            if this_bin['this_bin_is_empty']:
                continue
            x.append(this_bin['this_bin_boundary'][1])
            univariate = Univariate(this_bin['dependent_var_in_this_bin'])
            y.append(univariate.find_nearest_inverse_ecdf(boundary))
        x = np.array(x)
        y = np.array(y).T
        return bivariate_uncertainty_plot(x, y, boundary, scatter_ax, **kwargs)

    def out_put_mob_row_as_univariate_cdf_like(self, key, cdf_x: ndarray):
        cdf_y = Univariate(self.mob[key]['dependent_var_in_this_bin']).ecdf(cdf_x)
        cdf_like_ndarray = np.stack((cdf_y, cdf_x), axis=1)
        return UnivariatePDFOrCDFLike(cdf_like_ndarray=cdf_like_ndarray)

    # def out_put_mob_row_as_univariate_pdf_like(self, key, pdf_x: ndarray):
    #     pdf_y = Univariate(self.mob[key]['dependent_var_in_this_bin'])
    #     pdf_like_ndarray = np.stack((pdf_x, pdf_y), axis=1)
    #     return UnivariatePDFOrCDFLike(pdf_like_ndarray=pdf_like_ndarray)

    @staticmethod
    def sample_from_mob_fitting_like_dict(mob_fitting_like_dict: dict, number_of_sample_per_bin: int,
                                          *, jitter_of_predictor_var: bool = True,
                                          theoretic_min_value: float = np.NINF,
                                          theoretic_max_value: float = np.inf) -> Tuple[ndarray, ndarray]:
        """
        mob_fitting_like_dict的采样函数
        :param mob_fitting_like_dict:key是int索引，然后value是None或者dict，
        而且子dict中必须包含‘this_bin_boundary’和'this_bin_probability_model'字段
        :param number_of_sample_per_bin 每个bin的采样的样本数
        :param jitter_of_predictor_var 表示采样出来的predictor_var用一个固定的值(bin的中值)，还是一个在bin种均匀分布的随机值
        :param theoretic_min_value 理论最小值。所有采样bin的下界
        :param theoretic_max_value 理论最大值。所有采样bin的上界
        :return: 采样结果，x数组和y数组
        """
        x, y = [], []
        for this_bin in mob_fitting_like_dict.values():
            if this_bin['this_bin_is_empty']:
                continue
            # 这个bin的x的值
            if jitter_of_predictor_var:
                x.append(np.random.uniform(this_bin['this_bin_boundary'][0], this_bin['this_bin_boundary'][-1],
                                           number_of_sample_per_bin))
            else:
                x.append(np.full(number_of_sample_per_bin, this_bin['this_bin_boundary'][1]))
            # 这个bin的y的值
            if isinstance(this_bin['this_bin_probability_model'], GaussianMixture):
                this_bin_model = UnivariateGaussianMixtureModel(this_bin['this_bin_probability_model'],
                                                                theoretic_min_value=theoretic_min_value,
                                                                theoretic_max_value=theoretic_max_value)
                y.append(this_bin_model.sample(number_of_sample_per_bin))
            else:
                raise Exception("Unknown model")
        return np.array(x).flatten(), np.array(y).flatten()

    @staticmethod
    def plot_mob_fitting_like_uncertainty_scatter_by_sampling(mob_fitting_like_dict: dict,
                                                              number_of_sample_per_bin: int = 50000,
                                                              *, show_scatter: bool = True, **kwargs):
        x, y = MethodOfBins.sample_from_mob_fitting_like_dict(mob_fitting_like_dict, number_of_sample_per_bin)
        MethodOfBins(x, y).plot_mob_uncertainty(show_scatter, **kwargs)


class Bivariate(MethodOfBins):
    __slots__ = ('predictor_var_name', 'dependent_var_name',
                 'category')

    def __init__(self, predictor_var: ndarray, dependent_var: ndarray, **kwargs):
        """
        :param cannot_be_outlier_rule: 一系列的规则表示这些区域内的数据不应该是outlier。Example:
        (((x1_min, x1_max), (y1_min, y1_max)), ((x2_min, x2_max), (y2_min, y2_max)))表示：
        在x1_min→x1_max的情况下，如果数据对满足不小于y1_min且小于y1_max，则这对数据不是outlier，
        同时x2_min→x2_max的情况下，如果数据对满足不小于y2_min且小于y2_max，则这对数据不是outlier，
        以此类推，可有无限组exclude方案
        """
        super().__init__(predictor_var, dependent_var, **kwargs)
        self.predictor_var_name = kwargs.get('predictor_var_name', 'x')  # type: str
        self.dependent_var_name = kwargs.get('dependent_var_name', ' y')  # type: str
        self.category = kwargs.get('category')  # type: ndarray

    def __str__(self):
        return '{} instance ({} as predictor_var, {} as dependent_var) with {} recordings'.format(
            self.__class__.__name__,
            self.predictor_var_name,
            self.dependent_var_name,
            self.predictor_var.size)

    def plot_scatter(self, show_category: Tuple[int, ...] = None, **kwargs):
        if show_category is None:
            self.__fast_plot_scatter(**kwargs)
        elif isinstance(show_category, tuple):
            self.__plot_bivariate_classification_scatter(show_category=show_category, **kwargs)
        else:
            raise TypeError(r"'show_category' should be either Tuple[int, ...] or 'all'")

    def __fast_plot_scatter(self, **kwargs):
        """
        快速画二维散点图
        """
        scatter(self.predictor_var, self.dependent_var,
                x_label=self.predictor_var_name, y_label=self.dependent_var_name, **kwargs)

    def __plot_bivariate_classification_scatter(self, *, show_category: Tuple[int, ...], **kwargs):
        if self.category is None:
            raise Exception("Should specify 'category' attribute for Bivariate instance")
        bivariate_classification_scatter(self.predictor_var, self.dependent_var,
                                         x_label=self.predictor_var_name, y_label=self.dependent_var_name,
                                         category_ndarray=self.category,
                                         show_category=show_category,
                                         **kwargs)


class BivariateOutlier(Bivariate):
    __slots__ = ('mob_and_outliers_detection_considered_data_mask', 'cannot_be_outlier_rule', 'must_be_outlier_rule')

    def __init__(self, predictor_var: ndarray, dependent_var: ndarray,
                 *, bin_step: float = None,
                 mob_and_outliers_detection_considered_data_mask: ndarray = None,
                 cannot_be_outlier_rule: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = None,
                 must_be_outlier_rule: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = None,
                 **kwargs):
        super().__init__(predictor_var, dependent_var, bin_step=bin_step,
                         considered_data_mask_for_mob_calculation=mob_and_outliers_detection_considered_data_mask,
                         **kwargs)
        self.cannot_be_outlier_rule = cannot_be_outlier_rule
        self.must_be_outlier_rule = must_be_outlier_rule
        self.mob_and_outliers_detection_considered_data_mask = mob_and_outliers_detection_considered_data_mask

    def __rule_to_mask(self, rule):
        results = np.full(self.predictor_var.shape, False)
        for this_rule in rule:
            this_rule_mask_x = np.bitwise_and(self.predictor_var >= this_rule[0][0],
                                              self.predictor_var < this_rule[0][1])
            this_rule_mask_y = np.bitwise_and(self.dependent_var >= this_rule[1][0],
                                              self.dependent_var < this_rule[1][1])
            # 根据这条rule去更新cannot_be_outlier_results
            results[np.bitwise_and(this_rule_mask_x, this_rule_mask_y)] = True
        return results

    @property
    def cannot_be_outlier_mask(self) -> ndarray:
        if self.cannot_be_outlier_rule is None:
            raise Exception('Need some rules for excluding inliers')
        return self.__rule_to_mask(self.cannot_be_outlier_rule)

    @property
    def must_be_outlier_mask(self) -> ndarray:
        if self.must_be_outlier_rule is None:
            raise Exception('Need some rules for determining outliers')
        return self.__rule_to_mask(self.must_be_outlier_rule)

    def __modify_according_to_rule(self, outlier):
        if self.cannot_be_outlier_rule is not None:
            outlier[self.cannot_be_outlier_mask] = False  # 强制满足规则的数据对不是outlier
        if self.must_be_outlier_rule is not None:
            outlier[self.must_be_outlier_mask] = True  # 强制满足规则的数据对不是outlier
        return outlier

    def identify_shut_down_outlier(self, cannot_be_zero_predictor_var_range: tuple,
                                   zero_upper_tolerance_factor: float) -> ndarray:
        """
        找出不应该是0的dependent_var
        :return:
        """
        outlier = shut_down_outlier(predictor_var=self.predictor_var,
                                    dependent_var=self.dependent_var,
                                    cannot_be_zero_predictor_var_range=cannot_be_zero_predictor_var_range,
                                    zero_upper_tolerance_factor=zero_upper_tolerance_factor)
        outlier[~self.mob_and_outliers_detection_considered_data_mask] = False
        return outlier

    def identify_interquartile_outliers_based_on_method_of_bins(self) -> ndarray:
        """
        结合method of bins和interquartile方法去identify outliers
        这里没有mob_and_outliers_detection_considered_data_idx的原因时计算method_of_bins_obj的时候已经考虑了
        :return: outlier的布尔数组索引
        """
        outlier = self.identify_interquartile_outliers()
        outlier = self.__modify_according_to_rule(outlier)
        return outlier

    def __identify_outlier_using_sklearn(self, func: Callable, func_args: dict):
        outlier = np.full(self.predictor_var.shape, False)
        data = np.array([self.predictor_var[self.mob_and_outliers_detection_considered_data_mask],
                         self.dependent_var[self.mob_and_outliers_detection_considered_data_mask]]).T
        data_idx = np.where(self.mob_and_outliers_detection_considered_data_mask)[0]
        temp = func(data, data_idx, func_args)
        outlier[temp] = True
        outlier = self.__modify_according_to_rule(outlier)
        return outlier

    def identify_outlier_using_isolation_forest(self, isolationforest_args: dict = None):
        return self.__identify_outlier_using_sklearn(use_isolation_forest, isolationforest_args)

    def identify_outliers_using_local_outlier_factor(self, lof_args: dict = None):
        return self.__identify_outlier_using_sklearn(use_local_outlier_factor, lof_args)

    def identify_outliers_using_optics_maximum_size(self, optics_args: dict = None):
        return self.__identify_outlier_using_sklearn(use_optics_maximum_size, optics_args)
