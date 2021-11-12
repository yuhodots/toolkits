from toolkits.utils import load
from toolkits.cluster import sse


def main():
    path = '../dataset/classification300.npz'
    feature, classifier, label_f, label_c = load(path)
    err_arr = sse(feature, classifier, label_f, label_c, print_result=True)


if __name__ == '__main__':
    main()
