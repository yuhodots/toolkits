from toolkits.utils import load
from toolkits.cluster import nsse, batch_nsse


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    result = nsse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_nsse.csv')
    print(result, '\n')

    feature, classifier, label_f, label_c = load('../dataset/classification300_batch.npz')
    result = batch_nsse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_batch_nsse.csv')
    print(result, '\n')


if __name__ == '__main__':
    main()
