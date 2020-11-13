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
    # Depreciated. Use plot_from_uncertainty_like_dataframe instead
    """
    warnings.warn("Depreciated. Use plot_from_uncertainty_like_dataframe instead", DeprecationWarning)

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
                                         show_coverage_labels: StrOneDimensionNdarray = (),
                                         automatic_alpha_control: bool = False,
                                         **kwargs):
    """
    :param x: x-axis values
    :param uncertainty_like_dataframe: a pd.DataFrame, whose index is the str-type percentile and column is
    the index of recordings
    :param lower_half_percentiles: considered lower tails of percentiles
    :param ax:
    :param show_coverage_labels:
    :param automatic_alpha_control: This is designed to make the scatters in under layer (if they exist) with a constant
    visibility level. For example, before calling plot_from_uncertainty_like_dataframe, there may be a scatter plot,
    which is ax, and ax will be passed as an argument, and it is desired that plot_from_uncertainty_like_dataframe could
    provide variable alpha values for the fill_between for each pair of percentiles so that the scatters may have a
    constant visibility level.
    :return:
    """

    @creat_fig((5, 5 * 0.618), ax)
    def plot(_ax):
        nonlocal uncertainty_like_dataframe
        # type checking
        uncertainty_like_dataframe = UncertaintyDataFrame(uncertainty_like_dataframe)
        StrOneDimensionNdarray(lower_half_percentiles)
        # Infer the higher half percentiles as the inputs are only about lower tail
        higher_half_percentiles = uncertainty_like_dataframe.infer_higher_half_percentiles(lower_half_percentiles)
        # Get colour code
        cmap = cm.get_cmap(kwargs.pop('cmap_name') if 'cmap_name' in kwargs else 'bone')  # 'copper', 'jet', 'cool'
        norm = colors.Normalize(vmin=0, vmax=int(lower_half_percentiles.size))
        # For each pair of percentiles, add new plotting layer
        for i in range(lower_half_percentiles.size):
            this_lower_half_percentile = lower_half_percentiles[i]
            this_higher_half_percentile = higher_half_percentiles[i]
            # Prepare for label-adding, according to the requirements (i.e., 'show_coverage_labels')
            if lower_half_percentiles[i] in show_coverage_labels:
                this_coverage_label = ' '.join((f"{100 - 2 * float(this_lower_half_percentile):.0f}",
                                                '%'))
            else:
                this_coverage_label = None
            # If automatic_alpha_control is desired to infer the alpha value
            if automatic_alpha_control:
                alpha = np.linspace(0.5, 0, lower_half_percentiles.size)[i]
            else:
                alpha = kwargs.pop('alpha') if 'alpha' in kwargs else None
            # Add new plotting layer for this pair of percentiles
            _ax.fill_between(x,
                             uncertainty_like_dataframe(by_percentile=this_lower_half_percentile),
                             uncertainty_like_dataframe(by_percentile=this_higher_half_percentile),
                             facecolor=kwargs.pop('facecolor') if 'facecolor' in kwargs else cmap(1 - norm(i)),
                             edgecolor='k',
                             linewidth=0.25,
                             label=this_coverage_label,
                             alpha=alpha,
                             **kwargs
                             )
        # Add new plotting layer for mean value
        _ax = series(x, uncertainty_like_dataframe.iloc[-2].values,
                     color='fuchsia', linestyle='--', ax=_ax, label='Mean')
        _ax = series(x, uncertainty_like_dataframe(by_percentile=50),
                     color='orange', linestyle='-', ax=_ax, label='Median')
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
