# -*- coding:utf-8 -*-
"""
 Description: DBSCAN simple version implementation, the main sklearn memory is too high

@author: WangLeAi
@date: 2018/12/25
"""

import numpy as np
import getpass
from pathlib import Path
from File_Management.load_save_Func import save_pkl_file
from tqdm import tqdm

UNCLASSIFIED = False
NOISE = -1


def __dis(vector1, vector2):
    """
         Cosine angle
         :param vector1: Vector A
         :param vector2: Vector B
    :return:
    """
    distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    distance = max(0.0, 1.0 - float(distance))
    return distance


def __eps_neighborhood(vector1, vector2, eps):
    """
         Neighbor
         :param vector1: Vector A
         :param vector2: Vector B
         :param eps: maximum distance of the sample under the same domain
    :return:
    """
    return __dis(vector1, vector2) < eps


def __region_query(data, point_id, eps):
    """
         Core function
         :param data: data set, array
         :param point_id: core point
         :param eps: maximum distance of the sample under the same domain
    :return:
    """
    n_points = data.shape[0]
    seeds = []
    for i in range(0, n_points):
        if __eps_neighborhood(data[point_id, :], data[i, :], eps):
            seeds.append(i)
    return seeds


def __expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
    """
         Class cluster diffusion
         :param data: data set, array
         :param classifications: classification results
         :param point_id: current point
         :param cluster_id: classification category
         :param eps: maximum distance of the sample under the same domain
         :param min_points: minimum core points per cluster
    :return:
    """
    seeds = __region_query(data, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        mark = False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
        while len(seeds) > 0:
            current_point = seeds[0]
            results = __region_query(data, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        mark = True
    return mark


def dbscan_custom(data, eps, min_points):
    """
         Dbscan clustering
         :param data: data set, array
         :param eps: maximum distance of the sample under the same domain
         :param min_points: minimum core points per cluster
    :return:
    """
    cluster_id = 1
    n_points = data.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in tqdm(range(0, n_points), total=n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if __expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


if __name__ == "__main__":
    _data = np.loadtxt(Path(f"C:/Users/{getpass.getuser()}/OneDrive/PhD/01-PhDProject/02-Wind/MyProject/Data/Results/"
                            f"Filtering/TSE2020_dbscan_lof/Zelengrad WF DBSCAN INPUT.csv"),
                       delimiter=',')
    results = dbscan_custom(_data, 0.025, 5)
    save_pkl_file(Path("./Zelengrad WF DBSCAN_results.pkl"), results)
