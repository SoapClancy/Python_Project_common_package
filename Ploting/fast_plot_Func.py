from numpy import ndarray
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from typing import Union, Sequence
import datetime
import matplotlib.dates as mdates

from .utils import creat_fig, show_fig


@show_fig
def scatter(x: Union[ndarray, range], y: ndarray, ax=None, figure_size=(5, 5 * 0.618), rasterized=True, **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        # c = kwargs.pop('c') if 'c' in kwargs else 'b'
        s = kwargs.pop('s') if 's' in kwargs else 2
        color_bar_name = kwargs.pop('color_bar_name') if 'color_bar_name' in kwargs else False
        fig = _ax.scatter(x, y, s=s, rasterized=rasterized, **kwargs)
        if color_bar_name:
            cb = plt.colorbar(fig)
            cb.set_label(color_bar_name)
        return fig

    return plot


@show_fig
def scatter_density(x: ndarray, y: ndarray, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        nonlocal x
        nonlocal y
        valid_mask = np.bitwise_and(~np.isnan(x), ~np.isnan(y))
        x, y = x[valid_mask], y[valid_mask]
        nbins = 100
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[min(x):max(x):nbins * 1j, min(y):max(y):nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return _ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='jet', rasterized=True, **kwargs)

    return plot


@show_fig
# def series(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(8, 8 * 7 / 14.93),
def series(x: Union[range, ndarray, Sequence], y: Union[range, ndarray, Sequence] = None,
           ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        nonlocal y
        if y is None:
            y = np.arange(0, len(x))
            return _ax.plot(y, x, **kwargs)
        else:
            return _ax.plot(x, y, **kwargs)

    return plot


@show_fig
def step(x: Union[range, ndarray, Sequence], y: Union[range, ndarray, Sequence] = None,
         ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        nonlocal y
        if y is None:
            y = np.arange(0, len(x))
            return _ax.step(y, x, **kwargs)
        else:
            return _ax.step(x, y, **kwargs)

    return plot


@show_fig
def stem(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(5, 5 * 0.618), color='b', **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        nonlocal y
        kwargs.setdefault('markerfmt', ' ')
        kwargs.setdefault('basefmt', ' ')
        if y is None:
            y = np.arange(0, len(x))
            return _ax.stem(y, x, color, use_line_collection=True, **kwargs)
        else:
            return _ax.stem(x, y, color, use_line_collection=True, **kwargs)

    return plot


def time_series(x_axis_format="%y-%m-%d %H", tz=None, **kwargs):
    if not isinstance(kwargs['x'][0], datetime.datetime):
        raise Exception("time series的x值必须是datetime.datetime对象")

    x_lim = (kwargs['x'][0] - datetime.timedelta(seconds=1),
             kwargs['x'][-1] + datetime.timedelta(seconds=1))
    ax = series(figure_size=(10, 2.4), x_lim=x_lim, x_axis_format=x_axis_format, tz=tz, **kwargs)
    return ax


@show_fig
def bar(*args, ax=None, autolabel_format: str = None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        rects = _ax.bar(*args, **kwargs)
        if autolabel_format is not None:
            for rect in rects:
                height = rect.get_height()
                _ax.annotate(autolabel_format.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)
        return _ax

    return plot


@show_fig
def hist(hist_data: ndarray, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        return _ax.hist(x=hist_data, **kwargs)

    return plot


@show_fig
def vlines(x, ax=None, linewidth=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        ymin = kwargs.pop('ymin') if 'ymin' in kwargs else -10e2
        ymax = kwargs.pop('ymax') if 'ymax' in kwargs else 10e4
        linestyles = kwargs.pop('linestyles') if 'linestyles' in kwargs else '--'
        return _ax.vlines(x, ymin=ymin, ymax=ymax, linewidth=linewidth, linestyles=linestyles, alpha=0.95, **kwargs)

    return plot


@show_fig
def hlines(y, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        xmin = kwargs.pop('xmin') if 'xmin' in kwargs else -10e2
        xmax = kwargs.pop('xmax') if 'xmax' in kwargs else 10e4
        linestyles = kwargs.pop('linestyles') if 'linestyles' in kwargs else '--'
        return _ax.hlines(y, xmin=xmin, xmax=xmax, linestyles=linestyles, alpha=0.5, **kwargs)

    return plot


@show_fig
def matrix_plot(mat, ax=None, **kwargs):
    @creat_fig((5, 5), ax)
    def plot(_ax):
        return _ax.imshow(mat, **kwargs)

    return plot


@show_fig
def pcolormesh(x: ndarray, y: ndarray, color_value: ndarray,
               ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(size=figure_size, ax=ax)
    def plot(_ax):
        return _ax.pcolormesh(x, y, color_value, **kwargs)

    return plot


@show_fig
def acorr(x: ndarray, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(size=figure_size, ax=ax)
    def plot(_ax):
        return _ax.acorr(x, **kwargs)

    return plot
