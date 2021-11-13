from toolkits.utils import load
from toolkits.cluster import nsse, batch_nsse


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
    result = nsse(feature, classifier, label_f, label_c, print_result=True)
    print(result, '\n')

    feature, classifier, label_f, label_c = load('../dataset/classification300_batch.npz')
    result = batch_nsse(feature, classifier, label_f, label_c, print_result=True)
    print(result, '\n')


if __name__ == '__main__':
    main()
