from sklearn.ensemble import IsolationForest
from numpy import ndarray
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from Ploting.fast_plot_Func import *


def use_isolation_forest(data: ndarray, random_state=0, isolationforest_kwargs: dict = None, *, return_obj=False):
    isolationforest_kwargs = isolationforest_kwargs or {}
    clf = IsolationForest(max_samples=isolationforest_kwargs.get('max_samples', data.shape[0]),
                          random_state=random_state,
                          **isolationforest_kwargs)
    clf.fit(data)
    outlier = clf.predict(data) == -1
    # ax = scatter(*data[~outlier].T)
    # ax = scatter(*data[outlier].T, ax=ax)
    if return_obj:
        return clf
    else:
        return outlier


def use_local_outlier_factor(data: ndarray, data_idx: ndarray, lof_args: dict = None):
    data = StandardScaler().fit_transform(data)
    if lof_args is None:
        lof_args = {}
    lof = LocalOutlierFactor(**lof_args)
    outlier = lof.fit_predict(data) == -1
    return data_idx[outlier]


def use_optics_maximum_size(data: ndarray, optics_kwargs: dict = None):
    optics_kwargs = optics_kwargs or {}
    optics_labels = OPTICS(**optics_kwargs).fit_predict(data)
    maximum_size = -np.inf
    maximum_size_label = np.nan
    for this_label in np.unique(optics_labels):
        if sum(optics_labels == this_label) > maximum_size:
            maximum_size = sum(optics_labels == this_label)
            maximum_size_label = this_label
    outlier = optics_labels != maximum_size_label
    return outlier


def use_dbscan(data: ndarray, dbscan_kwargs: dict = None):
    dbscan_kwargs = dbscan_kwargs or {}
    dbscan_labels = DBSCAN(n_jobs=-1, algorithm='auto', **dbscan_kwargs).fit_predict(data)
    return dbscan_labels
