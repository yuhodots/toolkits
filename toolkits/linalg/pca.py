"""
Principal Component Analysis
"""
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict


def get_singular_values(feature: np.array, label: np.array) -> Dict:
    singular_values = dict()
    num_c = np.array(list(set(label))).shape[0]

    for idx in range(num_c):
        f_idx = np.where(label == idx)[0]
        pca = PCA()
        pca.fit(feature[f_idx])
        singular_values[idx] = list(pca.singular_values_)

    return singular_values


def get_sum_of_singular_values(feature: np.array, label: np.array) -> Dict:
    singular_values_dict = get_singular_values(feature, label)
    for key, value in singular_values_dict.items():
        singular_values_dict[key] = sum(value)
    return singular_values_dict


def get_average_sum_of_singular_values(feature: np.array, label: np.array) -> float:
    singular_values_dict = get_sum_of_singular_values(feature, label)

    val_sum = 0
    for value in singular_values_dict.values():
        val_sum += value

    return val_sum / len(singular_values_dict)
