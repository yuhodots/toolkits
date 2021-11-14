from toolkits.utils import load
from toolkits.pprint import pred


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    pred(feature, classifier, label_f, label_c, opt_save=True, path='../results/example_pred.csv')


if __name__ == '__main__':
    main()
