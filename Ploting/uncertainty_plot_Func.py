from matplotlib import pyplot as plt
from Ploting.fast_plot_Func import creat_fig, show_fig
from numpy import ndarray
from typing import Tuple
from matplotlib import cm, colors


@show_fig
def bivariate_uncertainty_plot(x: ndarray, y: ndarray, boundary: ndarray, ax=None):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        cmap = cm.get_cmap('bone')
        norm = colors.Normalize(vmin=0, vmax=int(boundary.size / 2))
        for i in range(0, int(boundary.size / 2), 1):
            coverage = 100 - int(100 * 2 * boundary[i])
            ax_.fill_between(x, y[i], y[boundary.size - i - 1], facecolor=cmap(1 - norm(i)), edgecolor='k',
                             linewidth=0.25, alpha=0.75,
                             label=str(coverage) + '%')
        ax_.legend(ncol=1)
        return ax_

    return plot


@show_fig
def series_uncertainty_plot(x: ndarray, y1: ndarray, y2: ndarray, ax=None,
                            *, facecolor, edgecolor, hatch: str, **kwargs):
    @creat_fig((10, 2.4), ax)
    def plot(ax_):
        ax_.fill_between(x, y1, y2, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, **kwargs)
        return ax_

    return plot
