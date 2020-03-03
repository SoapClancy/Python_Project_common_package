from sklearn.ensemble import IsolationForest
from numpy import ndarray
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import numpy as np


def use_isolation_forest(data: ndarray, data_idx: ndarray, isolationforest_args: dict = None):
    """
    :return: 返回outlier的索引
    """
    data = StandardScaler().fit_transform(data)
    if isolationforest_args is None:
        isolationforest_args = {}
    clf = IsolationForest(behaviour=isolationforest_args.get('behaviour', 'new'), **isolationforest_args)
    clf.fit(data)
    outlier = clf.predict(data) == -1
    return data_idx[outlier]


def use_local_outlier_factor(data: ndarray, data_idx: ndarray, lof_args: dict = None):
    """
    :return: 返回outlier的索引
    """
    data = StandardScaler().fit_transform(data)
    if lof_args is None:
        lof_args = {}
    lof = LocalOutlierFactor(**lof_args)
    outlier = lof.fit_predict(data) == -1
    return data_idx[outlier]


def use_optics_maximum_size(data: ndarray, data_idx: ndarray, optics_args: dict = None):
    """
    :return: 返回outlier的索引
    """
    data = StandardScaler().fit_transform(data)
    if optics_args is None:
        optics_args = {}
    optics_labels = OPTICS(**optics_args).fit_predict(data)
    maximum_size = -np.inf
    maximum_size_label = np.nan
    for this_label in np.unique(optics_labels):
        if sum(optics_labels == this_label) > maximum_size:
            maximum_size = sum(optics_labels == this_label)
            maximum_size_label = this_label
    outlier = optics_labels != maximum_size_label
    return data_idx[outlier]
