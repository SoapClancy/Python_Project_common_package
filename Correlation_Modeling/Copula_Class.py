from UnivariateAnalysis_Class import Univariate, UnivariateGaussianMixtureModel
from typing import Tuple, Iterable
import numpy as np
from numpy import ndarray
import matlab.engine
from matlab.mlarray import double
import matlab
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save, save_pkl_file, load_pkl_file
from sklearn.mixture import GaussianMixture
import warnings
from File_Management.path_and_file_management_Func import try_to_find_file
from Ploting.fast_plot_Func import scatter, hist, series, scatter_density
from python_project_common_path_Var import python_project_common_path_
from Data_Preprocessing.float_precision_control_Func import limit_ndarray_max_and_min
from Data_Preprocessing import float_eps
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike
import copy
from abc import ABCMeta, abstractmethod
from itertools import permutations
from typing import Union
from pathlib import Path

THREE_DIM_CVINE_CONSTRUCTION = ((1, 2), (1, 3), (2, 3, 1))
FOUR_DIM_CVINE_CONSTRUCTION = ((1, 2), (1, 3), (1, 4), (2, 3, 1), (2, 4, 1), (3, 4, 1, 2))


class Copula(metaclass=ABCMeta):
    __slots__ = ('ndarray_data', 'marginal_distribution', 'ndarray_data_in_uniform', 'str_name')
    pseudo_zero = float_eps * 100_000
    """
    Copula类，它的所有subclass都有有关marginal distribution的方法，而且都是用gmm去估计的
    """

    def __init__(self, ndarray_data: ndarray = None, *, ndarray_data_in_uniform=None,
                 marginal_distribution: Tuple[GaussianMixture, ...] = None,
                 marginal_distribution_file_: str = None, str_name: str = None):
        self.ndarray_data = ndarray_data  # type: ndarray
        # marginal_distribution的权限高于marginal_distribution_file_，如果它有值就直接用它，否则就查找文件/fit并存储
        if marginal_distribution is not None:
            self.marginal_distribution = marginal_distribution
        else:
            if marginal_distribution_file_ is not None:
                self.marginal_distribution = self.load_or_fit_marginal_distribution_by_gmm(marginal_distribution_file_)
        if ndarray_data_in_uniform is not None:
            self.ndarray_data_in_uniform = ndarray_data_in_uniform
        else:
            self.ndarray_data_in_uniform = self.__transform_ndarray_data_to_uniform_by_gmm()
        self.str_name = str_name

    def __str__(self):
        return self.str_name

    def plot_simulated(self, n: int = None, **kwargs):
        """
        只支持2维
        """
        if n is None:
            if self.ndarray_data_in_uniform is not None:
                n = int(self.ndarray_data_in_uniform.shape[0] * 1.2)
            else:
                n = 50_000
        sim = self.simulate(n)
        x_label = kwargs.pop('x_label') if 'x_label' in kwargs else 'sim_x1'
        y_label = kwargs.pop('y_label') if 'y_label' in kwargs else 'sim_x2'
        title = kwargs.pop('title') if 'title' in kwargs else self.str_name
        return scatter_density(sim[:, 0], sim[:, 1], x_label=x_label, y_label=y_label,
                               x_lim=(-0.02, 1.02), y_lim=(-0.02, 1.02), title=title, **kwargs)

    def plot_ndarray_data_in_uniform(self, **kwargs):
        """
        只支持2维
        """
        return scatter_density(self.ndarray_data_in_uniform[:, 0], self.ndarray_data_in_uniform[:, 1],
                               x_lim=(-0.02, 1.02), y_lim=(-0.02, 1.02), title=self.str_name,
                               x_label='measurements_x1', y_label='measurements_x2', **kwargs)

    def plot_ndarray_data_in_uniform_and_simulated(self, n: int = None, **kwargs):
        if n is None:
            n = int(self.ndarray_data_in_uniform.shape[0] * 1.2)
        self.plot_simulated(n=n, **kwargs)
        return self.plot_ndarray_data_in_uniform(**kwargs)

    def __transform_ndarray_data_to_uniform_by_gmm(self):
        """
        用gmm的cdf将ndarray_data的所有维度转换成uniform
        :return:
        """
        if self.ndarray_data is None:
            return None
        else:
            return self.transform_ndarray_data_like_to_uniform_by_gmm(self.ndarray_data)

    def limit_to_pseudo_zero_and_one(self, x: ndarray):
        return limit_ndarray_max_and_min(x, self.pseudo_zero, 1 - self.pseudo_zero)

    def transform_ndarray_data_like_to_uniform_by_gmm(self, ndarray_data_like: ndarray) -> ndarray:
        """
        用gmm的cdf将ndarray_data_like数据转成uniform。注意：如果不是初次建立模型，i.e.,
        用ndarray_data去fit marginal distribution，那么marginal distribution必须已经声明
        :param ndarray_data_like: shape[1]和初次建立模型时候的ndarray_data一样的数组
        :return:
        """
        uniform_ndarray_data_like = np.full(ndarray_data_like.T.shape, np.nan)
        for i, this_col_data in enumerate(ndarray_data_like.T):
            uniform_ndarray_data_like[i] = UnivariateGaussianMixtureModel(
                self.marginal_distribution[i]).cdf_estimate(this_col_data)
        # 将uniform的边界缩小，因为运算精度的问题。尤其是在和MATLAB混编的时候，eps不一样，经常出现singularity，巨坑
        uniform_ndarray_data_like = self.limit_to_pseudo_zero_and_one(uniform_ndarray_data_like)
        return uniform_ndarray_data_like.T

    def __fit_marginal_distribution_by_gmm(self) -> Tuple[GaussianMixture, ...]:
        """
        用gmm作为marginal distribution去拟合每一个维度
        :return: 每一个维度的gmm拟合结果，组成一个tuple
        """
        if np.any(np.isnan(self.ndarray_data)):
            warnings.warn("np.nan is found in 'ndarray_data'", UserWarning)
        if self.ndarray_data is None:
            raise Exception("Either specify 'marginal_distribution_file_' and load marginal model, or specify "
                            "'ndarray_data' and then fit")
        gmm_model = [Univariate(x[~np.isnan(x)]).fit_using_gaussian_mixture_model() for x in self.ndarray_data.T]
        return tuple(gmm_model)

    def load_or_fit_marginal_distribution_by_gmm(self, marginal_distribution_file_: Union[str, Path]) -> \
            Tuple[GaussianMixture, ...]:
        """
        尝试载入边缘分布模型，如果模型不存在，就fit，如果也给了路径，就fit and save
        :param marginal_distribution_file_: 模型载入/储存的路径
        :return: GaussianMixture组成的tuple
        """
        if marginal_distribution_file_ is not None:
            @load_exist_pkl_file_otherwise_run_and_save(Path(marginal_distribution_file_))
            def fit_marginal_distribution():
                if self.ndarray_data is None:
                    raise Exception("No valid model is detected in 'marginal_distribution_file_' and 'ndarray_data'")
                else:
                    return self.__fit_marginal_distribution_by_gmm()

            return fit_marginal_distribution()
        else:
            return self.__fit_marginal_distribution_by_gmm()

    def __find_linspace(self, linspace_number: int, linspace_based_on_actual: bool, col_idx, tile_number: int):
        if linspace_based_on_actual:
            true_output_max = UnivariateGaussianMixtureModel(
                self.marginal_distribution[col_idx]).find_nearest_inverse_cdf(1 - self.pseudo_zero)
            true_output_min = UnivariateGaussianMixtureModel(
                self.marginal_distribution[col_idx]).find_nearest_inverse_cdf(self.pseudo_zero)
            true_output_linspace = np.linspace(true_output_min, true_output_max, linspace_number)
            linspace_value = UnivariateGaussianMixtureModel(
                self.marginal_distribution[col_idx]).cdf_estimate(true_output_linspace)
        else:
            linspace_value = np.linspace(self.pseudo_zero, 1 - self.pseudo_zero, linspace_number)
            true_output_linspace = UnivariateGaussianMixtureModel(
                self.marginal_distribution[col_idx]).find_nearest_inverse_cdf(linspace_value)
        return np.tile(linspace_value, tile_number), np.tile(true_output_linspace, tile_number)

    def __cal_product_of_marginal_pdf(self, ndarray_data_like: ndarray):
        """
        计算边缘概率密度的乘积
        """
        marginal_pdf = np.full(ndarray_data_like.shape, np.nan)
        for this_margin_idx, this_margin in enumerate(self.marginal_distribution):
            if this_margin_idx >= ndarray_data_like.shape[1]:
                continue
            marginal_pdf[:, this_margin_idx] = UnivariateGaussianMixtureModel(this_margin).pdf_estimate(
                ndarray_data_like[:, this_margin_idx])
        return np.prod(marginal_pdf, axis=1).flatten()

    @abstractmethod
    def cal_copula_pdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None,
                       use_ndarray_data: bool = False, use_ndarray_data_in_uniform: bool = False):
        pass

    def cal_joint_pdf(self, *, ndarray_data_like: ndarray = None):
        """
        joint的pdf就是边缘pdf的乘积再乘上copula的pdf
        """
        product_of_marginal_pdf = self.__cal_product_of_marginal_pdf(ndarray_data_like)
        copula_pdf = self.cal_copula_pdf(ndarray_data_like=ndarray_data_like)
        return product_of_marginal_pdf * copula_pdf

    def cal_joint_pdf_with_uncertain_inputs(self, *, uncertain_ndarray_data_like: ndarray):
        pass

    @abstractmethod
    def cal_copula_cdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None):
        pass

    def cal_joint_cdf(self, *, ndarray_data_like: ndarray = None):
        """
        joint的cdf就是copula的cdf
        """
        return self.cal_copula_cdf(ndarray_data_like=ndarray_data_like)

    def cal_joint_cdf_with_uncertain_inputs(self, *, uncertain_ndarray_data_like: ndarray, uncertain_dims: tuple):
        pass

    @abstractmethod
    def simulate(self, n: int):
        pass

    def cal_conditional_pdf_unnormalised(self, *, ndarray_data_like: ndarray,
                                         linspace_number: int = 500,
                                         linspace_based_on_actual: bool = True):
        output_col_idx = int(np.argwhere(np.isnan(ndarray_data_like[0])))
        linspace_value, true_output_linspace = self.__find_linspace(linspace_number,
                                                                    linspace_based_on_actual,
                                                                    output_col_idx,
                                                                    ndarray_data_like.shape[0])
        ndarray_data_like_in_uniform = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)
        ndarray_data_like_in_uniform = np.repeat(ndarray_data_like_in_uniform, linspace_value.size, 0)
        ndarray_data_like_in_uniform[:, output_col_idx] = linspace_value

        pdf_full_pdf = self.cal_copula_pdf(ndarray_data_like_in_uniform=ndarray_data_like_in_uniform)
        return np.stack((pdf_full_pdf, true_output_linspace), axis=1)

    def cal_copula_cdf_partial_derivative(self, *, ndarray_data_like: ndarray = None,
                                          use_ndarray_data: bool = False,
                                          ndarray_data_like_in_uniform: ndarray = None,
                                          use_ndarray_data_in_uniform: bool = False,
                                          partial_derivative_var_idx: Tuple[int, ...],
                                          partial_derivative_delta: float = None):
        """
        copula类的通用方法。迫不得已。numerical求解。返回copula在给定位置的偏导数值
        """
        partial_derivative_delta = partial_derivative_delta or self.pseudo_zero
        # 用ndarray_data_like或者本来就有的ndarray_data
        if use_ndarray_data:
            ndarray_data_like = self.ndarray_data
        if use_ndarray_data_in_uniform:
            ndarray_data_like_in_uniform = self.ndarray_data_in_uniform
        if ndarray_data_like_in_uniform is None:
            ndarray_data_like_in_uniform = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)

        # 准备上界和下界的数据
        ndarray_data_like_in_uniform_plus = copy.deepcopy(ndarray_data_like_in_uniform)
        ndarray_data_like_in_uniform_plus[:, np.array(partial_derivative_var_idx)] += partial_derivative_delta

        ndarray_data_like_in_uniform_minus = copy.deepcopy(ndarray_data_like_in_uniform)
        ndarray_data_like_in_uniform_minus[:, np.array(partial_derivative_var_idx)] -= partial_derivative_delta

        # 计算上界和下界，并且用limit_ndarray_max_and_min防止奇点
        ndarray_data_like_in_uniform_mixed = np.concatenate((ndarray_data_like_in_uniform_plus,
                                                             ndarray_data_like_in_uniform_minus))
        ndarray_data_like_in_uniform_mixed = self.limit_to_pseudo_zero_and_one(ndarray_data_like_in_uniform_mixed)
        copula_conditional_cdf_mixed = self.cal_copula_cdf(
            ndarray_data_like_in_uniform=ndarray_data_like_in_uniform_mixed)
        if np.any(np.isnan(copula_conditional_cdf_mixed)):
            raise Exception('Numeric partial derivative fail!')
        mixed_boundary = int(ndarray_data_like_in_uniform_mixed.shape[0] / 2)
        true_delta = (ndarray_data_like_in_uniform_mixed[:mixed_boundary, np.array(partial_derivative_var_idx)] -
                      ndarray_data_like_in_uniform_mixed[mixed_boundary:,
                      np.array(partial_derivative_var_idx)]).flatten()
        result = (copula_conditional_cdf_mixed[:mixed_boundary] -
                  copula_conditional_cdf_mixed[mixed_boundary:]) / true_delta
        # 因为数值解有误差，所以要让所有值强制小于1，强制大于0
        result = self.limit_to_pseudo_zero_and_one(result)
        return result


class BivariateCopula(Copula):
    def cal_copula_pdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None,
                       use_ndarray_data: bool = False, use_ndarray_data_in_uniform: bool = False):
        super().cal_copula_pdf(ndarray_data_like=ndarray_data_like,
                               ndarray_data_like_in_uniform=ndarray_data_like_in_uniform,
                               use_ndarray_data=use_ndarray_data,
                               use_ndarray_data_in_uniform=use_ndarray_data_in_uniform)

    def cal_copula_cdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None):
        super().cal_copula_cdf(ndarray_data_like=ndarray_data_like,
                               ndarray_data_like_in_uniform=ndarray_data_like_in_uniform)

    def cal_joint_cdf_with_uncertain_inputs(self, *, uncertain_ndarray_data_like: ndarray, uncertain_dims: tuple):
        # 确定那个维度是带有uncertain的输入，哪个维度是带有certain (i.e., deterministic)的输入
        uncertain_dim = uncertain_dims[0]

        if uncertain_dim == 1:
            ndarray_data_like_with_lower_boundary = np.stack((uncertain_ndarray_data_like[:, 0, 0],
                                                              uncertain_ndarray_data_like[:, 1, 0]), axis=1)
            ndarray_data_like_with_upper_boundary = np.stack((uncertain_ndarray_data_like[:, 0, 0],
                                                              uncertain_ndarray_data_like[:, 1, 1]), axis=1)

            results = (self.cal_copula_cdf(
                ndarray_data_like=ndarray_data_like_with_upper_boundary) - self.cal_copula_cdf(
                ndarray_data_like=ndarray_data_like_with_lower_boundary))
        else:
            raise
        return results

    def simulate(self, n: int):
        super().simulate(n)


class GMCM(BivariateCopula):
    matlab_script_folder_path = python_project_common_path_.__str__() + r'\Correlation_Modeling\GMCM_MATLAB'

    __slots__ = ('gmcm_model_file_',)
    """
    ！！！这里的GMCM只考虑2元的情况！！！
    因为所有的GMCM底层代码都是基于MATLAB，所以初始化GMCM一定要声明gmcm_model_file_的路径，然后才可以正常与MATLAB交互
    """

    def __init__(self, *, gmcm_model_file_: str,
                 ndarray_data: ndarray = None,
                 ndarray_data_in_uniform: ndarray = None,
                 marginal_distribution: Tuple[GaussianMixture, ...] = None,
                 marginal_distribution_file_: str = None,
                 gmcm_fitting_k: int = 6, gmcm_max_fitting_iteration: int = 500, gmcm_fitting_attempt: int = 1,
                 str_name: str = None, ):
        super().__init__(ndarray_data, ndarray_data_in_uniform=ndarray_data_in_uniform,
                         marginal_distribution=marginal_distribution,
                         marginal_distribution_file_=marginal_distribution_file_,
                         str_name=str_name)
        self.gmcm_model_file_ = gmcm_model_file_
        if not try_to_find_file(gmcm_model_file_):
            self.__fit_gmcm_using_matlab(gmcm_fitting_k, gmcm_max_fitting_iteration, gmcm_fitting_attempt)

    def __fit_gmcm_using_matlab(self, gmcm_fitting_k: int, gmcm_max_fitting_iteration: int, gmcm_fitting_attempt: int):
        """
        用MATLAB拟合GMCM，并储存数据
        :param gmcm_fitting_k: GMCM的component数量
        :param gmcm_max_fitting_iteration: GMCM的max fitting iteration
        :return:
        """
        if gmcm_fitting_attempt == 1:
            eng = matlab.engine.start_matlab()
            eng.addpath(self.matlab_script_folder_path, nargout=0)
            eng.estimate_gmcm_and_save(double(self.ndarray_data_in_uniform.tolist()), self.gmcm_model_file_,
                                       gmcm_fitting_k,
                                       gmcm_max_fitting_iteration, nargout=0)
            eng.quit()
        else:
            for i in range(gmcm_fitting_attempt):
                save_name = self.gmcm_model_file_ + str(i) if i > 0 else self.gmcm_model_file_
                eng = matlab.engine.start_matlab()
                eng.addpath(self.matlab_script_folder_path, nargout=0)
                eng.estimate_gmcm_and_save(double(self.ndarray_data_in_uniform.tolist()),
                                           save_name,
                                           gmcm_fitting_k,
                                           gmcm_max_fitting_iteration, nargout=0)
                eng.quit()

    def __cal_gmcm_cdf_using_matlab(self, uniform_ndarray_data_like):
        eng = matlab.engine.start_matlab()
        eng.addpath(self.matlab_script_folder_path, nargout=0)
        gmcm_cdf_value = eng.load_gmcm_and_cal_cdf(double(uniform_ndarray_data_like.tolist()),
                                                   self.gmcm_model_file_, nargout=1)
        eng.quit()
        return np.asarray(gmcm_cdf_value).flatten()

    def __cal_gmcm_pdf_using_matlab(self, uniform_ndarray_data_like):
        eng = matlab.engine.start_matlab()
        eng.addpath(self.matlab_script_folder_path, nargout=0)
        gmcm_pdf_value = eng.load_gmcm_and_cal_pdf(double(uniform_ndarray_data_like.tolist()),
                                                   self.gmcm_model_file_, nargout=1)
        eng.quit()
        return np.asarray(gmcm_pdf_value).flatten()

    def __simulate_gmcm_using_matlab(self, n):
        eng = matlab.engine.start_matlab()
        eng.addpath(self.matlab_script_folder_path, nargout=0)
        simulated = eng.load_gmcm_and_simulate(self.gmcm_model_file_, n, nargout=1)
        eng.quit()
        return np.asarray(simulated)

    def cal_copula_pdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None,
                       use_ndarray_data: bool = False, use_ndarray_data_in_uniform: bool = False):
        if use_ndarray_data:
            ndarray_data_like = self.ndarray_data
        if use_ndarray_data_in_uniform:
            ndarray_data_like_in_uniform = self.ndarray_data_in_uniform
        if ndarray_data_like_in_uniform is None:
            ndarray_data_like_in_uniform = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)

        return self.__cal_gmcm_pdf_using_matlab(ndarray_data_like_in_uniform)

    def cal_copula_cdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None):
        if ndarray_data_like_in_uniform is None:
            uniform_ndarray_data_like = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)
        else:
            uniform_ndarray_data_like = ndarray_data_like_in_uniform
        return self.__cal_gmcm_cdf_using_matlab(uniform_ndarray_data_like)

    def simulate(self, n: int):
        return self.__simulate_gmcm_using_matlab(n)

    def cal_copula_cdf_partial_derivative(self, *, ndarray_data_like: ndarray = None,
                                          use_ndarray_data: bool = False,
                                          ndarray_data_like_in_uniform: ndarray = None,
                                          use_ndarray_data_in_uniform: bool = False,
                                          partial_derivative_var_idx: Tuple[int, ...],
                                          partial_derivative_delta: float = None):
        # 用ndarray_data_like或者本来就有的ndarray_data
        if use_ndarray_data:
            ndarray_data_like = self.ndarray_data
        if use_ndarray_data_in_uniform:
            ndarray_data_like_in_uniform = self.ndarray_data_in_uniform
        if ndarray_data_like_in_uniform is None:
            ndarray_data_like_in_uniform = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)
        eng = matlab.engine.start_matlab()
        eng.addpath(self.matlab_script_folder_path, nargout=0)
        cdf_partial_derivative_value = eng.load_gmcm_and_cdf_partial_derivative(
            double(ndarray_data_like_in_uniform.tolist()),
            partial_derivative_var_idx[0] + 1, self.gmcm_model_file_, nargout=1)
        eng.quit()
        result = np.asarray(cdf_partial_derivative_value).flatten()
        return self.limit_to_pseudo_zero_and_one(result)

    def cal_joint_cdf_with_uncertain_inputs(self, *, uncertain_ndarray_data_like: ndarray, uncertain_dims: tuple):
        return super().cal_joint_cdf_with_uncertain_inputs(uncertain_ndarray_data_like=uncertain_ndarray_data_like,
                                                           uncertain_dims=uncertain_dims)


class VineCopula(Copula):
    __slots__ = ('construction',)

    def __init__(self, ndarray_data: ndarray = None, *,
                 marginal_distribution_file_: str, construction: Tuple[tuple, ...]):
        super().__init__(ndarray_data, marginal_distribution_file_=marginal_distribution_file_)
        self.construction = self.resort_construction(construction)

    @staticmethod
    def resort_construction(construction):
        """
        按从小到大的顺序排列conditioned和conditioning变量，这是标准化操作，确保，例如，如果C12存在的话就不去白费功夫fit C21
        """
        resorted_construction = []
        for i in construction:
            this_edge_var = list(sorted(i[0:2]))
            conditioning = list(sorted(i[2:])) if (i.__len__() > 2) else []
            this_edge_var.extend(conditioning)
            resorted_construction.append(tuple(this_edge_var))
        return tuple(resorted_construction)

    @property
    def resolved_construction(self) -> dict:
        """
        解析construction，返回一个dict，key有两个：‘conditioned’和 ‘conditioning’，它们对应的都是tuple
        """
        result = {'conditioned': [], 'conditioning': []}
        for i in self.construction:
            result['conditioned'].append(tuple(sorted(i[0:2])))
            result['conditioning'].append(tuple(sorted(i[2:])) if (i.__len__() > 2) else ())
        result['conditioned'] = tuple(result['conditioned'])
        result['conditioning'] = tuple(result['conditioning'])
        return result

    @staticmethod
    def all_vars_valid_mask(x: ndarray):
        """
        两个变量都是有效值的mask
        """
        return np.all(~np.isnan(x), axis=1)

    def all_vars_valid_data(self, x: ndarray):
        """
        保证两个变量都是有效值
        """
        return x[self.all_vars_valid_mask(x), :]

    @property
    @abstractmethod
    def pair_copula_instance_of_each_edge(self) -> Tuple[Copula, ...]:
        pass

    # 索引模型树
    def identify_input_copula_for_tree(self, which_i) -> Tuple[int, int, int, int]:
        # 索引模型树idx
        def identify_input_copula_idx_for_tree(possible_models_idx: Tuple[tuple, ...]):
            for this_possible_model_idx in possible_models_idx:
                try:
                    input_copula_idx = self.construction.index(this_possible_model_idx)
                    return input_copula_idx
                except ValueError:
                    pass
            raise Exception("Catastrophic error! No model found!")

        # conditioning索引
        def model_idx_conditioning(input_copula_model_idx):
            to_div = {*self.resolved_construction['conditioning'][which_i]}.difference(
                {*self.resolved_construction['conditioning'][input_copula_model_idx]})
            if int(list(to_div)[0]) == self.construction[input_copula_model_idx][0]:
                conditioning_var_idx = 0
            else:
                conditioning_var_idx = 1
            return conditioning_var_idx

        model_idx_left = identify_input_copula_idx_for_tree(tuple(
            permutations((self.resolved_construction['conditioned'][which_i][0],
                          *self.resolved_construction['conditioning'][which_i]))))
        model_idx_left_conditioning = model_idx_conditioning(model_idx_left)

        model_idx_right = identify_input_copula_idx_for_tree(tuple(
            permutations((self.resolved_construction['conditioned'][which_i][1],
                          *self.resolved_construction['conditioning'][which_i]))))
        model_idx_right_conditioning = model_idx_conditioning(model_idx_left)

        return model_idx_left, model_idx_right, model_idx_left_conditioning, model_idx_right_conditioning

    def cal_copula_pdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None,
                       use_ndarray_data: bool = False, use_ndarray_data_in_uniform: bool = False) -> ndarray:
        """
        Vine的copula pdf需要按construction去递推分解
        """
        # 在这里，全部用的是ndarray_data_like_in_uniform
        if ndarray_data_like_in_uniform is None:
            ndarray_data_like_in_uniform = self.transform_ndarray_data_like_to_uniform_by_gmm(ndarray_data_like)
        pair_copula_of_each_edge_pdf = np.full((ndarray_data_like_in_uniform.shape[0],
                                                self.construction.__len__()), np.nan)

        initialised_pair_copula_of_each_edge = []
        for i, this_edge_copula in enumerate(self.pair_copula_instance_of_each_edge):
            # 如果这个pair_copula的conditioning变量时空集，i.e., 第一层的tree，或者说时root的时候，直接计算pdf
            # 另外一个很方便的一点是，self.resolved_construction['conditioned'][i][0]
            # 肯定小于self.resolved_construction['conditioned'][i][1]
            if self.resolved_construction['conditioning'][i].__len__() == 0:
                initialised_pair_copula_of_each_edge.append(this_edge_copula)
                initialised_pair_copula_of_each_edge[-1].ndarray_data_in_uniform = \
                    ndarray_data_like_in_uniform[:, [self.resolved_construction['conditioned'][i][0] - 1,
                                                     self.resolved_construction['conditioned'][i][1] - 1]]
                pair_copula_of_each_edge_pdf[:, i] = initialised_pair_copula_of_each_edge[-1].cal_copula_pdf(
                    use_ndarray_data_in_uniform=True)

            else:
                (input_left_copula_idx, input_right_copula_idx, left_conditioning,
                 right_conditioning) = self.identify_input_copula_for_tree(i)
                # 根据已经有的上层模型，计算左右输入
                input_left = initialised_pair_copula_of_each_edge[
                    input_left_copula_idx].cal_copula_cdf_partial_derivative(
                    use_ndarray_data_in_uniform=True, partial_derivative_var_idx=(left_conditioning,))
                input_right = initialised_pair_copula_of_each_edge[
                    input_right_copula_idx].cal_copula_cdf_partial_derivative(
                    use_ndarray_data_in_uniform=True, partial_derivative_var_idx=(right_conditioning,))
                # 将input_left和input_right当成这一层的输入写入
                initialised_pair_copula_of_each_edge.append(this_edge_copula)
                initialised_pair_copula_of_each_edge[-1].ndarray_data_in_uniform = np.stack((input_left,
                                                                                             input_right), axis=1)
                pair_copula_of_each_edge_pdf[:, i] = initialised_pair_copula_of_each_edge[-1].cal_copula_pdf(
                    use_ndarray_data_in_uniform=True)

        return np.prod(pair_copula_of_each_edge_pdf, axis=1)

    def cal_copula_cdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None):
        """
        这段代码待写。比较难。ref: https://github.com/MalteKurz/VineCopulaMatlab/issues/7
        """
        pass

    def simulate(self, n: int):
        pass


class VineGMCMCopula(VineCopula):
    __slots__ = ('gmcm_model_files_for_construction',)
    """
    Vine-GMCM模型，每一个pair都是GMCM
    construction：一个tuple，每个子元素也是tuple，子tuple的前两个元素代表conditioned变量，
    后面的元素(如果有的话)，代表conditioning变量
    gmcm_model_files_for_construction: 和construction的.__len__()一样，每一个对应的位置表示那个edge的GMCM的结果
    """

    def __init__(self, ndarray_data: ndarray = None, *, gmcm_model_folder_for_construction_path_: str, **kwargs):
        super().__init__(ndarray_data, **kwargs)
        self.gmcm_model_files_for_construction = self.__get_gmcm_model_files_for_construction(
            gmcm_model_folder_for_construction_path_)

    def __get_gmcm_model_files_for_construction(self, gmcm_model_folder_for_construction_path_):
        return tuple(
            [gmcm_model_folder_for_construction_path_ + '/GMCM_' + str(x) + '.mat' for x in self.construction])

    def fit(self):
        # fit模型
        initialised_pair_copula_of_each_edge = []
        for edge_idx, this_edge_gmcm in enumerate(self.gmcm_model_files_for_construction):
            # 没有的话就fit并且写入ndarray_data_like_in_uniform信息
            if not try_to_find_file(this_edge_gmcm):
                if self.resolved_construction['conditioning'][edge_idx] == ():
                    # edge_var_1 和 edge_var_2分别代表两个考虑的变量。至此，还是1代表第一维度，等下生成gmcm的时候的索引要减去1
                    edge_var_1, edge_var_2 = self.resolved_construction['conditioned'][edge_idx]
                    input_left = self.ndarray_data_in_uniform[:, edge_var_1 - 1]
                    input_right = self.ndarray_data_in_uniform[:, edge_var_2 - 1]
                else:
                    (input_left_copula_idx, input_right_copula_idx, left_conditioning,
                     right_conditioning) = self.identify_input_copula_for_tree(edge_idx)

                    # 计算左右输入
                    input_left = initialised_pair_copula_of_each_edge[
                        input_left_copula_idx].cal_copula_cdf_partial_derivative(
                        use_ndarray_data_in_uniform=True,
                        partial_derivative_var_idx=(left_conditioning,))

                    input_right = initialised_pair_copula_of_each_edge[
                        input_right_copula_idx].cal_copula_cdf_partial_derivative(
                        use_ndarray_data_in_uniform=True,
                        partial_derivative_var_idx=(right_conditioning,))

                initialised_pair_copula_of_each_edge.append(
                    GMCM(gmcm_model_file_=this_edge_gmcm,
                         ndarray_data_in_uniform=self.all_vars_valid_data(np.stack((input_left, input_right), axis=1)),
                         gmcm_fitting_k=8,
                         gmcm_max_fitting_iteration=3000,
                         gmcm_fitting_attempt=1,
                         str_name='GMCM_{}'.format(str(self.resolved_construction['conditioned'][edge_idx]) + '|' +
                                                   str(self.resolved_construction['conditioning'][edge_idx]))))
                # 重新修正ndarray_data_in_uniform使其包含nan，size对齐输入
                initialised_pair_copula_of_each_edge[-1].ndarray_data_in_uniform = np.stack(
                    (input_left, input_right), axis=1)
            # 有的话就初始化，为了下一层的可能没有fit模型做准备，即：得到模型和ndarray_data_like_in_uniform信息
            else:
                if self.resolved_construction['conditioning'][edge_idx] == ():
                    input_left = self.ndarray_data_in_uniform[
                                 :, self.resolved_construction['conditioned'][edge_idx][0] - 1]
                    input_right = self.ndarray_data_in_uniform[
                                  :, self.resolved_construction['conditioned'][edge_idx][1] - 1]
                else:
                    (input_left_copula_idx, input_right_copula_idx, left_conditioning,
                     right_conditioning) = self.identify_input_copula_for_tree(edge_idx)
                    # 计算这一层的模型拥有的ndarray_data_like_in_uniform
                    input_left = initialised_pair_copula_of_each_edge[
                        input_left_copula_idx].cal_copula_cdf_partial_derivative(
                        use_ndarray_data_in_uniform=True,
                        partial_derivative_var_idx=(left_conditioning,))
                    input_right = initialised_pair_copula_of_each_edge[
                        input_right_copula_idx].cal_copula_cdf_partial_derivative(
                        use_ndarray_data_in_uniform=True,
                        partial_derivative_var_idx=(right_conditioning,))
                initialised_pair_copula_of_each_edge.append(
                    GMCM(gmcm_model_file_=this_edge_gmcm,
                         ndarray_data_in_uniform=np.stack((input_left, input_right), axis=1),
                         str_name='GMCM_{}'.format(str(self.resolved_construction['conditioned'][edge_idx]) + '|' +
                                                   str(self.resolved_construction['conditioning'][edge_idx]))))
            # initialised_pair_copula_of_each_edge[-1].plot_simulated()

    @property
    def pair_copula_instance_of_each_edge(self):
        gmcm_models = []
        for edge_idx, this_edge_gmcm in enumerate(self.gmcm_model_files_for_construction):
            gmcm_models.append(GMCM(gmcm_model_file_=this_edge_gmcm,
                                    str_name='GMCM_{}'.format(
                                        str(self.resolved_construction['conditioned'][edge_idx]) + '|' +
                                        str(self.resolved_construction['conditioning'][edge_idx]))))
        return tuple(gmcm_models)

    def cal_copula_pdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None,
                       use_ndarray_data: bool = False, use_ndarray_data_in_uniform: bool = False):
        """
        直接按vine copula的方法来。如果不是因为copula中的cal_copula_pdf是@abstractmethod，那么这段代码可以略去...
        """
        return super().cal_copula_pdf(ndarray_data_like=ndarray_data_like,
                                      ndarray_data_like_in_uniform=ndarray_data_like_in_uniform,
                                      use_ndarray_data=use_ndarray_data,
                                      use_ndarray_data_in_uniform=use_ndarray_data_in_uniform)

    def cal_copula_cdf(self, *, ndarray_data_like: ndarray = None, ndarray_data_like_in_uniform: ndarray = None):
        """
        直接按vine copula的方法来。如果不是因为copula中的cal_copula_pdf是@abstractmethod，那么这段代码可以略去...
        """
        return super().cal_copula_cdf(ndarray_data_like=ndarray_data_like,
                                      ndarray_data_like_in_uniform=ndarray_data_like_in_uniform)

    def simulate(self, n: int):
        pass
