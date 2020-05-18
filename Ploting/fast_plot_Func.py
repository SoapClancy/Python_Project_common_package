from numpy import ndarray
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import cm
from typing import Union
import datetime
import matplotlib.dates as mdates
from io import BytesIO


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
                infer_y_lim_according_to_x_lim=False,
                x_ticks: tuple = None, y_ticks: tuple = None,
                save_file_: str = None,
                save_format: str = 'png',
                save_to_buffer: bool = False,
                legend_loc: str = 'best',
                legend_ncol: int = 1,
                **kwargs):
        ax = func(*args, **kwargs)
        if kwargs.get('label') is not None:
            ax.legend(loc=legend_loc, ncol=legend_ncol)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if isinstance(x_lim, tuple):
            plt.xlim(*x_lim)
            if all((infer_y_lim_according_to_x_lim,
                    x_lim[0] is not None,
                    x_lim[1] is not None)):  # 只支持部分函数，比如series()
                x = kwargs.get('x') if 'x' in kwargs else args[0]
                y = kwargs.get('y') if 'y' in kwargs else args[1]
                y_in_x_lim = y[np.argmin(np.abs(x - x_lim[0])):np.argmin(np.abs(x - x_lim[1]))]
                y_resolution = float(np.diff(np.sort(y_in_x_lim)[:2]))
                plt.ylim(np.min(y_in_x_lim) - y_resolution,
                         np.max(y_in_x_lim) + 0.1 * y_in_x_lim.max() - y_in_x_lim.min())
        if isinstance(y_lim, tuple):
            plt.ylim(*y_lim)
        if isinstance(x_ticks, tuple):
            plt.xticks(*x_ticks)
        if isinstance(y_ticks, tuple):
            plt.yticks(*y_ticks)
        plt.grid(True)
        # 如果要存入buffer
        if save_to_buffer:
            buf = BytesIO()
            plt.savefig(buf)
            plt.close()
            return buf
        else:
            plt.show()
        if all((isinstance(save_file_, str), isinstance(save_format, str))):
            plt.savefig(save_file_ + '.' + save_format, format=save_format, dpi=300)
        return plt.gca()

    return wrapper


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
def series(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        nonlocal y
        if y is None:
            y = np.arange(0, x.size)
            return _ax.plot(y, x, **kwargs)
        else:
            return _ax.plot(x, y, **kwargs)

    return plot


@show_fig
def stem(x: Union[range, ndarray], y: ndarray = None, ax=None, figure_size=(5, 5 * 0.618), **kwargs):
    @creat_fig(figure_size, ax)
    def plot(_ax):
        nonlocal y
        kwargs.setdefault('markerfmt', ' ')
        kwargs.setdefault('basefmt', ' ')
        if y is None:
            y = np.arange(0, x.size)
            return _ax.stem(y, x, 'b', use_line_collection=True, **kwargs)
        else:
            return _ax.stem(x, y, 'b', use_line_collection=True, **kwargs)

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
    def plot(_ax):
        return _ax.hist(x=hist_data, **kwargs)

    return plot


@show_fig
def vlines(x, ax=None, **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        ymin = kwargs.pop('ymin') if 'ymin' in kwargs else -10e2
        ymax = kwargs.pop('ymax') if 'ymax' in kwargs else 10e4
        linestyles = kwargs.pop('linestyles') if 'linestyles' in kwargs else '--'
        return _ax.vlines(x, ymin=ymin, ymax=ymax, linestyles=linestyles, alpha=0.5, **kwargs)

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
