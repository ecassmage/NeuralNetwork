import random


def to_int(arr):
    return [int(val) for val in arr]


def get_data(file_path_input, file_path_output, map_file):
    X = [to_int(line.strip().split(',')) for line in open(file_path_input)]
    y = [int(line.strip()) for line in open(file_path_output)]
    temp = list(zip(X, y))
    random.shuffle(temp)
    X, y = map(list, zip(*temp))
    X, y = condense_data(X, y, map_file)
    return list(X), list(y)


def split_into_groups(number_of_groups, number_in_group, data_inputs, data_outputs):
    groups_input = []
    groups_output = []
    for group in range(number_of_groups):
        groups_input.append(data_inputs[number_in_group*group:number_in_group*group+number_in_group])
        groups_output.append(data_outputs[number_in_group*group:number_in_group*group+number_in_group])
    return groups_input, groups_output, data_inputs[number_of_groups * number_in_group:], data_outputs[number_of_groups * number_in_group:]


def condense_data(data_inputs, data_outputs, map_file):
    import numpy as np
    weights = [int(val.strip()) for val in open(map_file)]
    n = np.array(data_inputs) / np.array(weights[1:])
    nn = np.array(data_outputs) / np.array(weights[0])
    return n.tolist(), nn.tolist()


def main():
    pass


if __name__ == '__main__':
    pass
