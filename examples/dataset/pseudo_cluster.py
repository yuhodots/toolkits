from sklearn.datasets import make_blobs
import numpy as np


class ClusterData(object):
    def __init__(
            self,
            n_sample=300,
            n_cluster=5,
            h_dim=64,
            seed=0
    ):

        self.n_sample = n_sample
        self.n_cluster = n_cluster
        self.h_dim = h_dim
        self.seed = seed
        self._make_data(n_sample, n_cluster, h_dim, seed)

    def __call__(self):
        return self.feature, self.prototype, self.label_f, self.label_p

    def _make_data(
            self,
            n_sample,
            n_cluster,
            h_dim,
            seed,
            cluster_std=10
    ):
        self.feature, self.label_f = make_blobs(n_samples=n_sample, centers=n_cluster,
                                                n_features=h_dim, random_state=seed, cluster_std=cluster_std)
        prototype = []
        label_p = []
        for i in range(max(self.label_f) + 1):
            idx = np.where(self.label_f == i)
            prototype.append(np.mean(self.feature[idx], axis=0))
            label_p.append(i)

        self.prototype = np.array(prototype)
        self.label_p = np.array(label_p)
