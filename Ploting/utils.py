from io import BytesIO
import functools
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

sns.set()


class BufferedFigureSaver(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_new_buffer(self, name: str, buffer: BytesIO):
        self.append([name, buffer])


def creat_fig(size: tuple, ax=None):
    def decorator(func):
        nonlocal ax
        if ax is None:
            plt.figure(figsize=size, constrained_layout=True)
            ax = plt.gca()
        func(ax)
        return ax

    return decorator


def show_fig(func):
    @functools.wraps(func)
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
                x_axis_format=None,
                tz=None,
                **kwargs):
        ax = func(*args, **kwargs)
        if kwargs.get('label') is not None:
            ax.legend(loc=legend_loc, ncol=legend_ncol, prop={'size': 10})
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
                plt.ylim(np.min(y_in_x_lim) - 5 * y_resolution,
                         np.max(y_in_x_lim) + y_resolution)
        if isinstance(y_lim, tuple):
            plt.ylim(*y_lim)
        if isinstance(x_ticks, tuple):
            plt.xticks(*x_ticks)
        if isinstance(y_ticks, tuple):
            plt.yticks(*y_ticks)
        plt.grid(True)
        # dates

        if isinstance(kwargs.get('x'), pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_format,
                                                              tz=tz))
        # 如果要存入buffer
        if save_to_buffer:
            buf = BytesIO()
            plt.savefig(buf)
            plt.close()
            return buf  # 返回buf
        else:
            plt.show()
        if all((isinstance(save_file_, str), isinstance(save_format, str))):
            plt.savefig(save_file_ + '.' + save_format, format=save_format, dpi=300)
        return plt.gca()  # 返回gca

    return wrapper
