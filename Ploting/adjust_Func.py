from matplotlib import pyplot as plt
from typing import Sequence, Iterable

LINESTYLE_STR = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

LINESTYLE_TUPLE = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'

    ('loosely dotted', (0, (1, 10))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),

    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def reassign_linestyles_recursively_in_ax(ax, simple_linestyles: bool = True):
    if simple_linestyles:
        linestyle_str_len = LINESTYLE_STR.__len__()
        for i in range(ax.lines.__len__()):
            ax.lines[i].set_linestyle(LINESTYLE_STR[i % linestyle_str_len][1])
    else:
        linestyle_tuple_len = LINESTYLE_TUPLE.__len__()
        for i in range(ax.lines.__len__()):
            ax.lines[i].set_linestyle(LINESTYLE_TUPLE[i % linestyle_tuple_len][1])
    ax.legend(prop={'size': 10})
    return ax


def adjust_legend_in_ax(ax, *, protocol=None, **kwargs):
    assert (protocol in (None, 'Outside center right'))
    kwargs.setdefault('ncol', 1)
    kwargs.setdefault('prop', {'size': 10})
    if protocol == 'Outside center right':
        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", **kwargs)
    else:
        kwargs.setdefault('loc', 'upper center')
        ax.legend(**kwargs)
    return ax


def adjust_legend_order_in_ax(ax, *, new_order_of_labels: Sequence[str]):
    handles, labels = ax.get_legend_handles_labels()
    assert (handles.__len__() == new_order_of_labels.__len__()), "Check 'new_order_of_labels' length"

    hl = sorted(zip(handles, labels),
                key=lambda x: [x[1] == y for y in new_order_of_labels],
                reverse=True)
    handles, labels = zip(*hl)
    ax.legend(handles, labels,
              ncol=ax.get_legend().__getattribute__('_ncol'),
              loc=ax.get_legend().__getattribute__('_loc'),
              prop={'size': 10})
    return ax


def adjust_lim_label_ticks(ax, **kwargs):
    for key, item in kwargs.items():
        if key == 'x_lim':
            func = ax.set_xlim
        elif key == 'y_lim':
            func = ax.set_ylim
        elif key == 'x_ticks':
            func = ax.set_xticks
        elif key == 'y_ticks':
            func = ax.set_yticks
        elif key == 'x_label':
            func = ax.set_xlabel
        elif key == 'y_label':
            func = ax.set_ylabel
        elif key == 'x_tick_labels':
            func = ax.set_xticklabels
        elif key == 'y_tick_labels':
            func = ax.set_yticklabels
        else:
            raise Exception("Unsupported keyword(s)")
        func(item)

    return ax
