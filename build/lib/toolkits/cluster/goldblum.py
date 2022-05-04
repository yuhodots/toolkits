import numpy as np


def rfc(
    feature: np.array,
    label: np.array,
) -> int:
    num_c = np.array(list(set(label))).shape[0]
    num_f = np.zeros(num_c)
    centroid = np.zeros([num_c, feature.shape[-1]])
    numerator = 0

    for idx in range(num_c):
        f_idx = np.where(label == idx)[0]
        num_f[idx] = f_idx.shape[0]
        centroid[idx] = np.mean(feature[f_idx], axis=0)
        numerator += np.sum((feature[f_idx] - centroid[idx]) ** 2)

    centroid_mean = np.mean(centroid, axis=0)
    denominator = np.sum((centroid - centroid_mean) ** 2)

    # Feature clustering regularizer (R_fc)
    # https://arxiv.org/pdf/2002.06753.pdf
    C = num_c           # the number of classes
    N = num_f.min()     # the number of data points per class
    assert np.all(num_f == num_f[0]), "The number of data point per class is not the same each other."
    rfc_score = (C * numerator) / (N * denominator)

    return rfc_score
