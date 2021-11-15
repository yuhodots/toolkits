"""
Simple print for predictions and true labels
"""

import numpy as np
import pandas as pd


def save_table(num_c, label_f, pred_arr, path):
    df_list = []
    for idx in range(num_c):
        f_idx = np.where(label_f == idx)[0]
        pred_list = list(np.array(pred_arr[f_idx], dtype=int))
        df_list.append([idx, len(pred_list), pred_list.count(idx), pred_list])

    df = pd.DataFrame(df_list, columns=['Class', 'N', 'Correct', 'Predict'])
    df.to_csv(path, index=False)


def print_table(num_c, label_f, pred_arr):
    table = "{:^9}|{:^9}|{:^9}|{:^9}\n".format('Class', 'N', 'Correct', 'Predict')
    table += "-" * 39 + "\n"
    for idx in range(num_c):
        f_idx = np.where(label_f == idx)[0]
        pred_list = list(np.array(pred_arr[f_idx], dtype=int))
        pred_str = ""
        for item in pred_list:
            pred_str += "{:^7}".format(str(item))
        table += "{:^9}|{:^9}|{:^9}|{}\n".format(idx, len(pred_list), pred_list.count(idx), pred_str)

    print(table)


def get_logit(feature, classifier, similarity):
    if similarity == 'euclidean':
        w = np.expand_dims(classifier, 0)
        q = np.expand_dims(feature, 1)
        logit = -np.sum(np.square(q - w), axis=2)
    else:
        assert False, "Not supported similarity"
    return logit


def pred(
        feature,
        classifier,
        label_f,
        label_c,
        similarity='euclidean',
        opt_save=False,
        path=None,
):
    num_c = np.array(list(set(label_c))).shape[0]
    num_f = feature.shape[0]
    pred_arr = np.zeros(num_f)

    for idx in range(num_c):
        f_idx = np.where(label_f == idx)[0]
        logit = get_logit(feature[f_idx], classifier, similarity)
        pred_arr[f_idx] = np.argmax(logit, axis=1)

    print_table(num_c, label_f, pred_arr)

    if opt_save:
        assert path is not None, "Please enter the save path."
        save_table(num_c, label_f, pred_arr, path)
