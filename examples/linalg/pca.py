from toolkits.utils import load
from toolkits.linalg import get_singular_values, get_sum_of_singular_values, get_average_sum_of_singular_values


def main():
    feature, classifier, label_f, label_c = load('../dataset/classification300.npz')

    print("1. get_singular_values")
    print(get_singular_values(feature, label_f), '\n')

    print("2. get_sum_of_singular_values")
    print(get_sum_of_singular_values(feature, label_f), '\n')

    print("3. get_average_sum_of_singular_values")
    print(get_average_sum_of_singular_values(feature, label_f))


if __name__ == '__main__':
    main()
