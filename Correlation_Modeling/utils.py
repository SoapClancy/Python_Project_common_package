from scipy.stats import pearsonr, spearmanr, kendalltau
from BivariateAnalysis_Class import Bivariate
from typing import Callable
from ConvenientDataType import OneDimensionNdarray, ndarray
import numpy as np
from typing import Union
from Ploting.fast_plot_Func import *


class CorrelationFuncMapperMeta(type):
    def __getitem__(self, item):
        if item == 'Pearson':
            return pearsonr
        elif item == 'Spearman':
            return spearmanr
        elif item == 'Kendall':
            return kendalltau
        else:
            raise Exception("Unknown correlation function")


class CorrelationFuncMapper(metaclass=CorrelationFuncMapperMeta):
    pass


class CorrelationAnalyser:
    def __init__(self, *args, **kwargs):
        pass


class BivariateCorrelationAnalyser(CorrelationAnalyser, Bivariate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(CorrelationAnalyser, self).__init__(*args, **kwargs)

    def __call__(self, correlation_coefficient_name: str, *args, **kwargs):
        func = CorrelationFuncMapper[correlation_coefficient_name]
        return func(self.predictor_var,
                    self.dependent_var, *args, **kwargs)


class AutoCorrelationAnalyser:
    def __new__(cls, x: Union[OneDimensionNdarray, ndarray]):
        x = OneDimensionNdarray(x)
        return BivariateCorrelationAnalyser(x[:-1], x[1:])


if __name__ == '__main__':
    tt = np.arange(1, 11)
    cc = AutoCorrelationAnalyser(tt)
