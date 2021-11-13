"""
Sum of squared errors (SSE)
"""

import numpy as np


def sse_per_class(feature, classifier):
    return np.sum(np.sum(np.power(classifier - feature, 2), axis=1), axis=0)


def sse(feature, classifier, label_f, label_c, print_result=False):
    num_c = label_c.shape[0]
    num_f = np.zeros(num_c)
    err_arr = np.zeros(num_c)

    for idx in range(num_c):
        f_idx = np.where(label_f == label_c[idx])[0]
        err_arr[idx] = sse_per_class(feature[f_idx], classifier[idx])
        num_f[idx] = f_idx.shape[0]

    avg_err = np.sum(err_arr) / num_c

    if print_result:
        table = "|{:^7}|{:^8}|{:^7}|\n".format('Class', 'SSE', 'N')
        table += "-" * 26 + "\n"
        for idx in range(num_c):
            table += "|{:^7}|{:^8}|{:^7}|\n".format(idx, np.round(err_arr[idx], 2), int(num_f[idx]))
        table += "-" * 26 + "\n"
        table += "Average SSE: {}".format(np.round(avg_err, 2))
        print(table)

    return avg_err


def batch_sse(feature, classifier, label_f, label_c, print_result=False):

    assert len(label_c.shape) >= 2, "The input form is not batch."

    num_batch = label_c.shape[0]
    num_c = label_c.shape[1]
    num_f = np.zeros((num_batch, num_c))
    err_arr = np.zeros((num_batch, num_c))

    for i in range(num_batch):
        for idx in range(num_c):
            f_idx = np.where(label_f[i] == label_c[i, idx])[0]
            err_arr[i, idx] = sse_per_class(feature[i, f_idx], classifier[i, idx])
            num_f[i, idx] = f_idx.shape[0]

    num_f = np.mean(num_f, axis=0)
    err_arr = np.mean(err_arr, axis=0)
    avg_err = np.sum(err_arr) / num_c

    if print_result:
        table = "|{:^7}|{:^8}|{:^7}|\n".format('Class', 'SSE', 'N avg')
        table += "-" * 26 + "\n"
        for idx in range(num_c):
            table += "|{:^7}|{:^8}|{:^7}|\n".format(idx, np.round(err_arr[idx], 2), int(num_f[idx]))
        table += "-" * 26 + "\n"
        table += "Average SSE: {}".format(np.round(avg_err, 2))
        print(table)

    return avg_err
