from toolkits.utils import load
from toolkits.cluster import rfc


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    result = rfc(feature, label_f)
    print(result, '\n')


if __name__ == '__main__':
    main()
