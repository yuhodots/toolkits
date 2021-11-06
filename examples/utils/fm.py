import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from examples.dataset.pseudo_cluster import ClusterData
from toolkits.utils import save, load
import pprint


def main():
    data = ClusterData()
    feature, classifier, label_f, label_c = data()
    path = '../results/data.npz'
    save(feature, classifier, label_f, label_c, path)
    feature, classifier, label_f, label_c = load(path)

    result = {
        "feature": feature.shape,
        "classifier": classifier.shape,
        "label_f": label_f.shape,
        "label_c": label_c.shape,
    }
    pprint.pprint(result)


if __name__ == '__main__':
    main()
