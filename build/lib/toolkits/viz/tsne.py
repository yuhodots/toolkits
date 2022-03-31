"""
Visualize features and classifiers using t-SNE plot.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def make_colors(y_set):
    n_label = len(y_set)
    np.random.seed(42)
    colors = [[np.random.choice(range(256)) / 256,
               np.random.choice(range(256)) / 256,
               np.random.choice(range(256)) / 256]
              for _ in range(n_label)]
    return colors


def tsne(feature, classifier, label_f, label_c, save_path, perplexity=30, seed=42):
    """ t-SNE plot for features and classifiers
    :param feature: Feature vectors(n_data, dim)
    :param classifier : Classifier vectors (n_classes, dim)
    :param label_f : Label array of input feature vectors
    :param label_c:  Label array of input classifier vectors
    :param save_path: Save path of t-SNE result
    :param perplexity: Perplexity
    :param seed: Random seed
    """
    model = TSNE(random_state=seed, perplexity=perplexity)

    x = np.concatenate([feature, classifier])
    y = np.concatenate([label_f, label_c])
    y_set = list(set(y))
    colors = make_colors(y_set)

    out = model.fit_transform(x)

    for i in range(feature.shape[0]):
        c_idx = y_set.index(y[i])
        plt.scatter(out[i, 0], out[i, 1], color=colors[c_idx], s=10)

    for i in range(classifier.shape[0]):
        idx = feature.shape[0] + i
        plt.scatter(out[idx, 0], out[idx, 1], color='blue', s=70, marker='X')
        plt.text(out[idx, 0], out[idx, 1], str(int(y[idx])), fontdict={'size': 13, 'color': 'white', 'fontweight':1000})
        plt.text(out[idx, 0], out[idx, 1], str(int(y[idx])), fontdict={'size': 13})

    size = 2.5 * np.std(out)
    plt.xlim(-size, size)
    plt.ylim(-size, size)

    plt.show()
    plt.savefig(save_path)
    plt.close()
