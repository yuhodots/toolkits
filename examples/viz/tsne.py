import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from examples.dataset.pseudo_cluster import ClusterData
from toolkits.viz import tsne


def main():
    data = ClusterData()
    feature, classifier, label_f, label_c = data()
    save_path = '../results/tsne.png'
    tsne(feature, classifier, label_f, label_c, save_path)


if __name__ == '__main__':
    main()
