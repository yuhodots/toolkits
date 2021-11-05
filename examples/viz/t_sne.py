import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from examples.dataset.pseudo_cluster import ClusterData
from toolkits.viz.t_sne import t_sne


def main():
    data = ClusterData()
    feature, classifier, label_f, label_c = data()
    save_path = '../results/t_sne.png'
    t_sne(feature, classifier, label_f, label_c, save_path)


if __name__ == '__main__':
    main()
