"""
Visualize features and classifiers using t-SNE plot.
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def make_colors(y_set):
    n_label = len(y_set)
    colors = [[np.random.choice(range(256)) / 256,
               np.random.choice(range(256)) / 256,
               np.random.choice(range(256)) / 256]
              for _ in range(n_label)]
    return colors


def t_sne(feature, classifier, label_f, label_c, save_path, perplexity=30, seed=42):

    model = TSNE(random_state=seed, perplexity=perplexity)

    x = np.concatenate([feature, classifier])
    y = np.concatenate([label_f, label_c])
    y_set = list(set(y))
    colors = make_colors(y_set)

    out = model.fit_transform(x)

    for i in range(feature.shape[0]):
        c_idx = y_set.index(y[i])
        plt.text(out[i, 0], out[i, 1], str(int(y[i])), fontdict={'size': 6}, color='grey')
        plt.scatter(out[i, 0], out[i, 1], color=colors[c_idx], s=10)

    for i in range(classifier.shape[0]):
        idx = feature.shape[0] + i
        c_idx = y_set.index(y[idx])
        plt.text(out[idx, 0], out[idx, 1], str(int(y[idx])), fontdict={'size': 6}, color='grey')
        plt.scatter(out[idx, 0], out[idx, 1], color=colors[c_idx], s=50, marker=(5, 2))

    plt.xlim(out[:, 0].min(), out[:, 0].max())
    plt.ylim(out[:, 1].min(), out[:, 1].max())

    plt.show()
    plt.savefig(save_path)
    plt.close()
