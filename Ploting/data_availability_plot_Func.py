from Ploting.fast_plot_Func import *
import pandas as pd
import numpy as np
from Ploting.adjust_Func import *


def data_availability_plot(data: pd.DataFrame, name_mapper: dict = None, **kwargs):
    assert isinstance(data.index, pd.DatetimeIndex)

    # Rename
    name_mapper = name_mapper or {}
    name_mapper_inverse = {val: key for key, val in name_mapper.items()}
    data = data.rename(mapper=name_mapper, axis=1)

    ax = kwargs.get('ax')
    for i, column_name in enumerate(data.columns):
        available_mask = ~np.isnan(data[column_name].values)
        x = data.index[available_mask].values
        ax = scatter(x, np.full(x.shape[0], i),
                     s=30, marker="|", color="g", ax=ax, **kwargs)
    ax.set_xlabel(kwargs.get("x_label", "Timestamp"), fontsize=10)
    ax.set_yticks(list(range(data.columns.__len__())))
    ax.set_yticklabels(data.columns.tolist(), fontdict={'fontsize': 10})
    # Rename back
    data = data.rename(mapper=name_mapper_inverse, axis=1)
    return ax
