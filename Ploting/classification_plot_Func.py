from matplotlib import pyplot as plt
from Ploting.utils import creat_fig, show_fig
from numpy import ndarray
from typing import Tuple


@show_fig
def bivariate_classification_scatter(x: ndarray, y: ndarray, ax=None, *,
                                     category_ndarray: ndarray,
                                     show_category: Tuple[int, ...],
                                     show_category_color: Tuple[str, ...] = None,
                                     show_category_marker: Tuple[str, ...] = None,
                                     show_category_size: Tuple[int, ...] = None,
                                     show_category_label: Tuple[str, ...] = None,
                                     alpha: float = None,
                                     **kwargs):
    @creat_fig((5, 5 * 0.618), ax)
    def plot(ax_):
        legend_handle_list = []
        for i, this_show_category in enumerate(show_category):
            # this_data
            this_x = x[category_ndarray == this_show_category]
            this_y = y[category_ndarray == this_show_category]
            # color
            if show_category_color is not None:
                this_show_category_color = show_category_color[i]
            else:
                this_show_category_color = None
            # marker
            if show_category_marker is not None:
                this_show_category_marker = show_category_marker[i]
            else:
                this_show_category_marker = None
            # size
            if show_category_size is not None:
                this_show_category_size = show_category_size[i]
            else:
                this_show_category_size = 2
            # label
            if show_category_label is not None:
                this_show_category_label = show_category_label[i]
            else:
                this_show_category_label = None
            this_scatter_handle = ax_.scatter(this_x, this_y,
                                              c=this_show_category_color,
                                              s=this_show_category_size,
                                              rasterized=True,
                                              label=this_show_category_label,
                                              marker=this_show_category_marker,
                                              alpha=alpha,
                                              **kwargs)
            legend_handle_list.append(this_scatter_handle)
        if show_category_label:
            plt.legend(handles=legend_handle_list, loc='upper left', ncol=1)

    return plot
