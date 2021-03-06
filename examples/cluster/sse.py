from toolkits.utils import load
from toolkits.cluster import sse, batch_sse


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    result = sse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_sse.csv')
    print(result, '\n')

    feature, classifier, label_f, label_c = load('../dataset/classification300_batch.npz')
    result = batch_sse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_batch_sse.csv')
    print(result, '\n')


if __name__ == '__main__':
    main()
