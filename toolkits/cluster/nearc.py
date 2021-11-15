"""
top-N nearest interfering centroid
"""

import numpy as np


def print_table(result):
    num_c = result.shape[0]
    N = result.shape[1]

    table = "{:^9}".format("Class")
    for i in range(N):
        table += "|{:^7}".format("Near{}".format(i+1))
    table += "\n" + "-" * (N * 10)

    for idx in range(num_c):
        table += "\n{:^9}|".format(idx)
        for i in range(N):
            table += "{:^8}".format(int(result[idx, i]))

    print(table)


def get_nearest(N, idx, centroids):

    anchor = centroids[idx]
    dist = np.sum(np.square(centroids - anchor), axis=1)

    result = []
    for i in range(N):
        top_n_idx = np.argsort(dist)[i+1]     # exclude itself (exclude 0th result)
        result.append(top_n_idx)

    return result


def nearc(feature, label, N=5, opt_print=False):
    num_c = np.array(list(set(label))).shape[0]
    num_f = np.zeros(num_c)
    centroids = np.zeros((num_c, feature.shape[-1]))

    result = np.zeros((num_c, N))

    for idx in range(num_c):
        f_idx = np.where(label == idx)[0]
        centroids[idx] = np.mean(feature[f_idx], axis=0)

    for idx in range(num_c):
        f_idx = np.where(label == idx)[0]
        result[idx] = get_nearest(N, idx, centroids)
        num_f[idx] = f_idx.shape[0]

    if opt_print:
        print_table(result)

    return result