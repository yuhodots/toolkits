from toolkits.utils import load
from toolkits.pprint import pred_summary


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    pred_summary(feature, classifier, label_f, label_c, opt_save=True, path='../results/example_pred.csv')


if __name__ == '__main__':
    main()
