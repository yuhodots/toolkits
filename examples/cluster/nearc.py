from toolkits.utils import load
from toolkits.cluster import nearc


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    nearc(feature, label_f, 5, opt_print=True)


if __name__ == '__main__':
    main()
