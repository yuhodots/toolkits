"""
Extract the log between the two input sentences
"""
import os


def _file2list(path):
    """ Get file content to list type
    :param path: file path
    :return: file content list
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return lines


def _between_lines(lines, beg, end, single=True):
    """ Extract lines from `beg` to `end`
    :param lines: content list which is split for each newline
    :param beg: starting point of parsing
    :param end: end point of parsing
    :param single: whether there are multiple sections to exist
    :return: extracted lines
    """

    # Initialize variables
    start_idx = -1
    end_idx = -1
    indices = []

    # Parsing
    for idx, line in enumerate(lines):
        if line.startswith(beg):
            start_idx = idx
        if line.startswith(end):
            end_idx = idx

        if start_idx != -1 and end_idx != -1:
            assert end_idx >= start_idx, "must be 'end_idx >= start_idx'"
            indices.append([start_idx, end_idx])

            if single:
                break

    return [lines[item[0]:item[1]] for item in indices]


def between_lines(input_str, beg, end, single=True):
    """ Extract lines from `beg` to `end`
    :param input_str: input string
    :param beg: starting point of parsing
    :param end: end point of parsing
    :param single: whether there are multiple sections to exist
    :return: extracted lines
    """
    lines = input_str.splitlines()
    return _between_lines(lines, beg, end, single)


def between_lines_on_file(file_path, beg, end, single=True):
    """ Extract lines from `beg` to `end` from the target file
    :param file_path: file path
    :param beg: starting point of parsing
    :param end: end point of parsing
    :param single: whether there are multiple sections to exist
    :return: extracted lines
    """
    lines = _file2list(file_path)
    return _between_lines(lines, beg, end, single)


def between_lines_on_dir(dir_path, beg, end, single=True, path_filter=None):
    """ Extract lines from `beg` to `end` from the target directory
    :param dir_path: directory path
    :param beg: starting point of parsing
    :param end: end point of parsing
    :param single: whether there are multiple sections to exist
    :param path_filter: search only files containing `path_filter`
    :return: dictionary (key: file path, value: extracted lines)
    """
    file_list = os.listdir(dir_path)

    if path_filter:
        file_list = [item for item in file_list if path_filter in item]

    file_list = [os.path.join(dir_path, item) for item in file_list]

    file2lines = dict()

    for file_path in file_list:
        file2lines[file_path] = between_lines_on_file(file_path, beg, end, single)

    return file2lines
