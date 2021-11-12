"""
Sum of squared errors (SSE)
"""

import numpy as np
import pprint


def sse_per_class(feature, classifier):
    return np.sum(np.sum(np.power(classifier - feature, 2), axis=1), axis=0)


def sse(feature, classifier, label_f, label_c, print_result=False):
    num_c = label_c.shape[0]
    err_arr = np.zeros(num_c)

    for idx in range(num_c):
        f_idx = np.where(label_f == label_c[idx])
        err_arr[idx] = sse_per_class(feature[f_idx], classifier[idx])

    if print_result:
        result = {}
        for idx in range(num_c):
            result[label_c[idx]] = err_arr[idx]
        pprint.pprint(result)

    return err_arr
