from windrose import WindroseAxes
import matplotlib.cm as cm
from Ploting.fast_plot_Func import *
import pandas as pd
import numpy as np
from Ploting.adjust_Func import *
from Ploting.utils import creat_fig


def wind_rose_plot(ws, wd):
    both_not_nan_mask = np.bitwise_and(~np.isnan(ws), ~np.isnan(wd))
    ws, wd = ws[both_not_nan_mask], wd[both_not_nan_mask]
    ax = WindroseAxes.from_ax(theta_labels=["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"])
    ax.bar(wd, ws, normed=True, opening=0.95, edgecolor='white')
    ax.set_legend(loc='best')
    # plt.ion()
    # plt.show()
    # plt.draw()
    # plt.pause(0.001)
