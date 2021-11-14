"""
Sum of squared errors (SSE)
"""

import numpy as np
import pandas as pd


def save_table(num_c, num_f, err_arr, path):
    df_list = []
    for idx in range(num_c):
        df_list.append([idx, np.round(err_arr[idx], 2), num_f[idx]])

    df = pd.DataFrame(df_list, columns=['Class', 'SSE', 'N'])
    df.to_csv(path, index=False)


def print_table(num_c, num_f, err_arr, title):
    avg_err = np.sum(err_arr) / num_c

    table = "|{:^30}|\n".format(title)
    table += "|{:^9}|{:^10}|{:^9}|\n".format('Class', 'SSE', 'N')
    table += "-" * 32 + "\n"
    for idx in range(num_c):
        table += "|{:^9}|{:^10}|{:^9}|\n".format(idx, np.round(err_arr[idx], 2), num_f[idx])
    table += "-" * 32 + "\n"
    table += "Average SSE: {}".format(np.round(avg_err, 2))

    print(table)


def sse_per_class(feature, centroid):
    return np.sum(np.sum(np.power(centroid - feature, 2), axis=1), axis=0)


def sse(
        feature,
        label,
        opt_print=False,
        opt_save=False,
        path=None,
):
    num_c = np.array(list(set(label))).shape[0]
    num_f = np.zeros(num_c)
    err_arr = np.zeros(num_c)

    for idx in range(num_c):
        f_idx = np.where(label == idx)[0]
        centroid = np.mean(feature[f_idx], axis=0)
        err_arr[idx] = sse_per_class(feature[f_idx], centroid)
        num_f[idx] = f_idx.shape[0]

    avg_err = np.sum(err_arr) / num_c

    if opt_print:
        print_table(num_c, num_f, err_arr, '`toolkits.cluster.sse`')
    if opt_save:
        assert path is not None, "Please enter the save path."
        save_table(num_c, num_f, err_arr, path)

    return avg_err


def batch_sse(
        feature,
        label,
        opt_print=False,
        opt_save=False,
        path=None,
):

    assert len(label.shape) >= 2, "The input form is not batch."

    num_batch = label.shape[0]
    num_c = np.array(list(set(label[0]))).shape[0]
    num_f = np.zeros((num_batch, num_c))
    err_arr = np.zeros((num_batch, num_c))

    for i in range(num_batch):
        for idx in range(num_c):
            f_idx = np.where(label[i] == idx)[0]
            centroid = np.mean(feature[i, f_idx], axis=0)
            err_arr[i, idx] = sse_per_class(feature[i, f_idx], centroid)
            num_f[i, idx] = f_idx.shape[0]

    num_f = np.mean(num_f, axis=0)
    err_arr = np.mean(err_arr, axis=0)
    avg_err = np.sum(err_arr) / num_c

    if opt_print:
        print_table(num_c, num_f, err_arr, '`toolkits.cluster.batch_sse`')
    if opt_save:
        assert path is not None, "Please enter the save path."
        save_table(num_c, num_f, err_arr, path)

    return avg_err
