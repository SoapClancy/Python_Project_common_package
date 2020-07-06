from matplotlib import pyplot as plt
from Ploting.utils import creat_fig, show_fig
from numpy import ndarray
from typing import Tuple
from matplotlib import cm, colors
import pandas as pd
import numpy as np
from ConvenientDataType import StrOneDimensionNdarray, UncertaintyDataFrame
import warnings
from Ploting.fast_plot_Func import *


@show_fig
def bivariate_uncertainty_plot(x: ndarray, y: ndarray, boundary: ndarray, ax=None):
    """
    #TODO Depreciated. Use plot_from_uncertainty_like_dataframe instead
    """

    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        warnings.warn("推荐使用plot_from_uncertainty_like_dataframe 函数", DeprecationWarning)
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
def plot_from_uncertainty_like_dataframe(x: ndarray,
                                         uncertainty_like_dataframe: UncertaintyDataFrame,
                                         lower_half_percentiles: StrOneDimensionNdarray,
                                         ax=None, *,
                                         show_coverage_labels: StrOneDimensionNdarray = None):
    """
    :param x: x-axis values
    :param uncertainty_like_dataframe: a pd.DataFrame, whose index is the str-type percentile and column is
    the index of recordings
    :param lower_half_percentiles: considered lower tails of percentiles
    :param ax:
    :param show_coverage_labels
    :return:
    """

    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        nonlocal uncertainty_like_dataframe
        # type checking
        uncertainty_like_dataframe = UncertaintyDataFrame(uncertainty_like_dataframe)
        StrOneDimensionNdarray(lower_half_percentiles)
        #
        higher_half_percentiles = uncertainty_like_dataframe.infer_higher_half_percentiles(lower_half_percentiles)
        cmap = cm.get_cmap('bone')
        norm = colors.Normalize(vmin=0, vmax=int(lower_half_percentiles.size))
        for i in range(lower_half_percentiles.size):
            this_lower_half_percentile = lower_half_percentiles[i]
            this_higher_half_percentile = higher_half_percentiles[i]
            if show_coverage_labels is not None:
                if lower_half_percentiles[i] in show_coverage_labels:
                    this_coverage_label = ' '.join((f"{100 - 2 * float(this_lower_half_percentile):.0f}",
                                                    '%'))
                else:
                    this_coverage_label = None
            else:
                this_coverage_label = None
            _ax.fill_between(x,
                             uncertainty_like_dataframe.loc[this_lower_half_percentile].values,
                             uncertainty_like_dataframe.loc[this_higher_half_percentile].values,
                             facecolor=cmap(1 - norm(i)),
                             edgecolor='k',
                             linewidth=0.25,
                             label=this_coverage_label,
                             )
        _ax = series(x, uncertainty_like_dataframe.iloc[-1].values,
                     color=(0, 1, 0), linestyle='--', ax=_ax, label='Mean')
        # _ax.legend(ncol=1)

        return _ax

    return plot


@show_fig
def series_uncertainty_plot(x: ndarray, y1: ndarray, y2: ndarray, ax=None,
                            *, facecolor, edgecolor, hatch: str, **kwargs):
    @creat_fig((10, 2.4), ax)
    def plot(_ax):
        _ax.fill_between(x, y1, y2, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, **kwargs)
        return _ax

    return plot
