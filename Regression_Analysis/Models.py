import torch
from torch import nn
from pyro.nn import PyroModule
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
from Ploting.fast_plot_Func import *
import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist
import warnings
from pyro.infer import MCMC, NUTS


class BayesianRegression(PyroModule):
    def __init__(self, in_features: int,
                 out_features: int, *,
                 fit_intercept: bool = True,
                 weight_prior: dist.Distribution = None,
                 bias_prior: dist.Distribution = None):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features, bias=fit_intercept)
        # 自动生成weight_prior
        if weight_prior is None:
            warnings.warn("weight_prior未设定，已经自动设置成 ~ Normal(0, 100)（多半失败）", UserWarning)
            weight_prior = dist.Normal(0., 100.).expand(torch.Size([out_features, in_features])).to_event(2)
        self.linear.weight = PyroSample(weight_prior)
        # 不一定需要bias
        if fit_intercept:
            # 自动生成bias_prior
            if bias_prior is None:
                warnings.warn("bias_prior未设定，已经自动设置成 ~ Normal(0, 100)（多半失败）", UserWarning)
                bias_prior = dist.Normal(0., 100.).expand(torch.Size([out_features])).to_event(1)
            self.linear.bias = PyroSample(bias_prior)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # TODO sigma分布
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

    def run_mcmc(self, x: torch.Tensor, y: torch.Tensor, *,
                 num_samples: int = 1000,
                 warmup_steps: int = 200) -> MCMC:
        nuts_kernel = NUTS(self)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(x, y)
        return mcmc

    @staticmethod
    def predict(mcmc_instance_object: MCMC, x: torch.Tensor) -> ndarray:
        samples = mcmc_instance_object.get_samples()
        pred = []
        for i in range(samples['sigma'].shape[0]):
            pred.append(np.dot((x.numpy()), samples['linear.weight'][i].T))
            if 'linear.bias' in samples:
                pred[-1] += float(samples['linear.bias'][i])
        return np.array(pred)

    @staticmethod
    def ols_results(x: torch.Tensor, y: torch.Tensor, fit_intercept: bool) -> tuple:
        ols_reg = LinearRegression(fit_intercept=fit_intercept).fit(x.numpy(), y.numpy())
        return ols_reg.coef_, ols_reg.intercept_

    @staticmethod
    def lasso_results(x: torch.Tensor, y: torch.Tensor, fit_intercept: bool) -> tuple:
        lasso_reg = Lasso(fit_intercept=fit_intercept).fit(x.numpy(), y.numpy())
        return lasso_reg.coef_, lasso_reg.intercept_
