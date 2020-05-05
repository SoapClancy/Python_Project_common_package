from numpy import ndarray
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import cm
from typing import Union
import datetime
import matplotlib.dates as mdates


def creat_fig(size: tuple, ax=None):
    def wrapper(func):
        nonlocal ax
        if ax is None:
            ax = plt.figure(figsize=size, constrained_layout=True)
            ax = plt.gca()
        func(ax)
        return ax

    return wrapper


def show_fig(func):
    def wrapper(*args, title: str = None,
                x_label: str = None, y_label: str = None,
                x_lim: tuple = None, y_lim: tuple = None,
                x_ticks: tuple = None, y_ticks: tuple = None,
                save_file_: str = None,
                save_format: str = 'png',
                legend_loc: str = 'best',
                ncol: int = 1,
                **kwargs):
        ax = func(*args, **kwargs)
        if kwargs.get('label') is not None:
            ax.legend(loc=legend_loc, ncol=ncol)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if isinstance(x_lim, tuple):
            plt.xlim(*x_lim)
        if isinstance(y_lim, tuple):
            plt.ylim(*y_lim)
        if isinstance(x_ticks, tuple):
            plt.xticks(*x_ticks)
        if isinstance(y_ticks, tuple):
            plt.yticks(*y_ticks)
        plt.grid(True)
        plt.show()
        if all((isinstance(save_file_, str), isinstance(save_format, str))):
            plt.savefig(save_file_ + '.' + save_format, format=save_format, dpi=300)
        return plt.gca()

    return wrapper


@show_fig
def scatter(x: Union[ndarray, range], y: ndarray, ax=None, figure_size=(5, 5 * 0.618), rasterized=True, **kwargs):
    @creat_fig(figure_size, ax)
    def plot(ax_):
        # c = kwargs.pop('c') if 'c' in kwargs else 'b'
        s = kwargs.pop('s') if 's' in kwargs else 2
        color_bar_name = kwargs.pop('color_bar_name') if 'color_bar_name' in kwargs else False
        fig = ax_.scatter(x, y, s=s, rasterized=rasterized, **kwargs)
        if color_bar_name:
            cb = plt.colorbar(fig)
            cb.set_label(color_bar_name)
        return fig

    return plot


@show_fig
def scatter_density(x: ndarray, y: ndarray, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        nonlocal x
        nonlocal y
        valid_mask = np.bitwise_and(~np.isnan(x), ~np.isnan(y))
        x, y = x[valid_mask], y[valid_mask]
        nbins = 100
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[min(x):max(x):nbins * 1j, min(y):max(y):nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return ax_.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='jet', rasterized=True, **kwargs)

    return plot


@show_fig
def series(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(ax_):
        nonlocal y
        if y is None:
            y = np.arange(0, x.size)
            return ax_.plot(y, x, **kwargs)
        else:
            return ax_.plot(x, y, **kwargs)

    return plot


@show_fig
def stem(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(ax_):
        nonlocal y
        kwargs.setdefault('markerfmt', ' ')
        kwargs.setdefault('basefmt', ' ')
        if y is None:
            y = np.arange(0, x.size)
            return ax_.stem(y, x, use_line_collection=True, **kwargs)
        else:
            return ax_.stem(x, y, use_line_collection=True, **kwargs)

    return plot


def time_series(x_axis_format=None, tz=None, **kwargs):
    if not isinstance(kwargs['x'][0], datetime.datetime):
        raise Exception("time series的x值必须是datetime.datetime对象")

    x_lim = (kwargs['x'][0] - datetime.timedelta(seconds=1),
             kwargs['x'][-1] + datetime.timedelta(seconds=1))
    ax = series(figure_size=(10, 2.4), x_lim=x_lim, **kwargs)
    if x_axis_format:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_format, tz=tz))
    return ax


@show_fig
def hist(hist_data: ndarray, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        return ax_.hist(x=hist_data, **kwargs)

    return plot


@show_fig
def vlines(x, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        ymin = kwargs.pop('ymin') if 'ymin' in kwargs else -10e2
        ymax = kwargs.pop('ymax') if 'ymax' in kwargs else 10e4
        linestyles = kwargs.pop('linestyles') if 'linestyles' in kwargs else '--'
        return ax_.vlines(x, ymin=ymin, ymax=ymax, linestyles=linestyles, alpha=0.5, **kwargs)

    return plot


@show_fig
def hlines(y, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        xmin = kwargs.pop('xmin') if 'xmin' in kwargs else -10e2
        xmax = kwargs.pop('xmax') if 'xmax' in kwargs else 10e4
        linestyles = kwargs.pop('linestyles') if 'linestyles' in kwargs else '--'
        return ax_.hlines(y, xmin=xmin, xmax=xmax, linestyles=linestyles, alpha=0.5, **kwargs)

    return plot


@show_fig
def matrix_plot(mat, ax=None, **kwargs):
    @creat_fig((5, 5), ax)
    def plot(ax_):
        return ax_.imshow(mat, **kwargs)

    return plot
