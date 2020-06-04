import torch
from torch import nn
from pyro.nn import PyroModule
from sklearn.linear_model import LinearRegression
import numpy as np
from Ploting.fast_plot_Func import *
import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist


class BayesianRegression(PyroModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand(torch.Size([out_features, in_features])).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand(torch.Size([out_features])).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
