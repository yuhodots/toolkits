
from toolkits.parse import between_lines, between_lines_on_dir, between_lines_on_file


def main():
    list_data = between_lines_on_file('../dataset/document1.txt', beg="Organizational", end="Finance", single=True)
    for item in list_data:
        print("\n".join(item))

    dict_data = between_lines_on_dir('../dataset', beg="Organizational", end="Finance", single=True,
                                     path_filter="document")
    for k, v in dict_data.items():
        print(k)
        for item in v:
            print("\n".join(item))


if __name__ == '__main__':
    main()
