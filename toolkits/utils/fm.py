"""
File management tools.
"""

import pickle as pkl
import numpy as np


def save(feature, classifier, label_f, label_c, file_path):
    """Save vectors ans labels"""
    file_format = file_path.split('.')[-1]

    if file_format == 'pkl':
        data = {"feature": feature, "classifier": classifier,
                "label_f": label_f, "label_c": label_c}
        with open(file_path, "wb") as f:
            pkl.dump(data, f)
    elif file_format == 'npz':
        np.savez(file_path, feature=feature, classifier=classifier, label_f=label_f, label_c=label_c)
    else:
        assert False, "Not supported data format."


def load(file_path):
    """Load vectors ans labels from data file"""
    file_format = file_path.split('.')[-1]

    if file_format == 'pkl':
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            feature = data['feature']
            classifier = data['classifier']
            label_f = data['label_f']
            label_c = data['label_c']
    elif file_format == 'npz':
        data = np.load(file_path)
        feature = data['feature']
        classifier = data['classifier']
        label_f = data['label_f']
        label_c = data['label_c']
    else:
        assert False, "Not supported data format."

    return feature, classifier, label_f, label_c
