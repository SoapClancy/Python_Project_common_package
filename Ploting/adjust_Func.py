from matplotlib import pyplot as plt


def reassign_linestyles_recursively_in_ax(ax, simple_linestyles: bool = True):
    linestyle_str = [
        ('solid', 'solid'),  # Same as (0, ()) or '-'
        ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
        ('dashed', 'dashed'),  # Same as '--'
        ('dashdot', 'dashdot')]  # Same as '-.'

    linestyle_tuple = [
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

    if simple_linestyles:
        linestyle_str_len = linestyle_str.__len__()
        for i in range(ax.lines.__len__()):
            ax.lines[i].set_linestyle(linestyle_str[i % linestyle_str_len][1])
    else:
        linestyle_tuple_len = linestyle_tuple.__len__()
        for i in range(ax.lines.__len__()):
            ax.lines[i].set_linestyle(linestyle_tuple[i % linestyle_tuple_len][1])
    ax.legend()
    return ax


def adjust_legend_in_ax(ax, *, protocol=None, **kwargs):
    kwargs.setdefault('ncol', 1)
    if protocol == 'Outside center left':
        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", **kwargs)
    else:
        kwargs.setdefault('loc', 'upper center')
        ax.legend(**kwargs)
    return ax
