import numpy as np
import sys


def save_network(location, network):
    file = open(location, 'w')
    np.set_printoptions(threshold=sys.maxsize, precision=25)
    for layer in network.layers:
        file.write(f"{layer.weights.shape}\n")
        file.write(str(layer.weights.tolist()) + '\n')
        file.write(str(layer.biases.tolist()) + '\n\n')
    file.close()
    pass


def load_network(location, n):
    file = open(location)
    arr = ''.join([line.strip().replace('\n', '') for line in file])
    splitted = arr.replace(']]', '').replace('(', '[[').replace(')', '').split('[[')
    layer = False
    left, right = 0, 0
    weights = []
    biases = []

    weight = []
    for lineUp in splitted:
        pass
        for line in lineUp.split('], ['):
            if line == '':
                continue

            if not layer:
                layer = True
                left, right = line.split(', ')
                left, right = int(left), int(right)
            else:
                if left == 0:
                    layer = False
                    biases.append([[float(val) for val in line.split(', ')]])
                    weights.append(weight)
                    weight = []
                else:
                    weight.append([float(val) for val in line.split(', ')])
                    left -= 1
    n.load(weights, biases, len(biases[-1][0]))

    file.close()
    return n


def save_arr(location, arr):
    np.set_printoptions(threshold=sys.maxsize, precision=25)
    file = open(location, 'w')
    file.write(str(arr))
    file.write('\n')
    file.close()


def main():
    # arr = np.array([[1.21312312312313, 2, 3], [4, 5, 6]])
    # save_arr('file.txt', arr)
    load_network("neural_save_0.txt")


if __name__ == "__main__":
    main()
    pass
