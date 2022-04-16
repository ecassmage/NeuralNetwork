import numpy as np
import sys
import copy

import data_management
from logs import log

mk_log = True

if mk_log:
    logger = log.Log('log_test.txt')


class NeuralNetwork:
    def __init__(self, data_inputs, hidden_layer, data_outputs, scalar=3):
        self.output_basic = data_outputs
        self.outputs, self.classes = self.create_matrix(data_outputs, hidden_layer[-1])
        self.inputs = self.check_for_matrix(data_inputs)
        self.hidden_layer = hidden_layer

        self.layers = [Layer(hidden_layer[num], layer, Activation()) for num, layer in enumerate(hidden_layer[1:])]
        # self.loss = None
        self.current_layer = 0
        self.scalar = scalar

        self.error = 0

    def predict_input(self, input_data_set):
        self.inputs = self.check_for_matrix(input_data_set)
        self.run_network()
        val = self.get_predictions()
        return val

    def change_inputs_outputs(self, new_inputs, new_outputs):
        self.inputs = self.check_for_matrix(new_inputs)
        self.outputs = self.create_matrix(new_outputs)
        self.output_basic = new_outputs
        self.current_layer = 0

    def get_predictions(self):
        return np.argmax(self.get_output(), axis=1)

    def run_network(self):

        input_data = self.inputs
        self.current_layer = 0

        for num, layer in enumerate(self.layers):
            if num != 0: self.current_layer += 1
            layer.forward(input_data)
            input_data = layer.get_post_output()

    def train_network(self, n, debug=False):

        for _ in range(n):
            self.run_network()
            if debug:
                print(self.get_output())
                print(self.calculate_sum_loss())
                print(f"Node chosen: {self.get_predictions()}")
                print('\n')
            if mk_log:
                logger.write(f"Losses -> {self.calculate_sum_loss()}")
                logger.write(f"Predictions          -> {self.get_predictions()}")
                logger.write(f"Correct Predictions  -> {('[' + ' '.join([str(int(val)) for val in self.output_basic]) + ']')}")
                r, w = check_accuracy(self.get_predictions(), self.output_basic)
                logger.write(f"right: {r}, wrong: {w}", end='\n\n')

            self.loss()
        if debug:
            print(f"odd error: {self.error}")

    def shell_network(self, new_weights, biases):
        networks = [
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # + +
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # + -
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # - +
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # - -
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # + =
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # - =
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs),  # = +
            NeuralNetwork(self.inputs, self.hidden_layer, self.outputs)   # = -
        ]

        for network in networks:
            for num, layer in enumerate(network.layers):
                layer.weights = copy.deepcopy(self.layers[num].weights)
                layer.biases = copy.deepcopy(self.layers[num].biases)
                layer.activation = copy.deepcopy(self.layers[num].activation)
                layer.output = copy.deepcopy(self.layers[num].output)

        for current, network in enumerate(networks):
            for num, dj in enumerate(reversed(tuple(zip(new_weights, biases)))):

                if current in [0, 1, 4]:
                    network.layers[num].weights = network.layers[num].weights + (network.scalar * dj[0])
                elif current in [2, 3, 5]:
                    network.layers[num].weights = network.layers[num].weights - (network.scalar * dj[0])

                if current in [0, 2, 6]:
                    network.layers[num].biases = network.layers[num].biases + (dj[1])
                elif current in [1, 3, 7]:
                    network.layers[num].biases = network.layers[num].biases - (dj[1])

        for network in networks:
            network.run_network()

        neural_cost = self.calculate_sum_loss()
        arr = [neural_cost - network.calculate_sum_loss() for network in networks]

        # neural_pos_better_cost = neural_cost - new_network_pos.calculate_sum_loss()
        # neural_neg_better_cost = neural_cost - new_network_neg.calculate_sum_loss()

        if max(arr) < 0:
            print("Whoa!!!", arr)
            # self.scalar /= 1.5

        index = arr.index(max(arr))
        for num, dj in enumerate(reversed(tuple(zip(new_weights, biases)))):
            match index:
                case 0:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * dj[0])
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * dj[1])
                    pass
                case 1:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * dj[0])
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * dj[1])
                    pass
                case 2:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * dj[0])
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * dj[1])
                    pass
                case 3:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * dj[0])
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * dj[1])
                    pass
                case 4:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * dj[0])
                    pass
                case 5:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * dj[0])
                    pass
                case 6:
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * dj[1])
                case 7:
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * dj[1])
                case _:
                    exit()
        # if arr[0] > arr[2]:
        #     return '+'
        # elif arr[2] > arr[0]:
        #     return '-'
        # else:
        #     return '='
        # # if new_network_pos. - cost
        pass

    def calculate_sum_loss(self):
        b = (self.get_output() - self.outputs) ** 2
        return np.sum(b)

    def loss(self):

        dJdW = []
        biases = []

        delta_input = -(self.outputs - self.get_output()) * 2
        dJdW_input = self.layers[-2].get_post_output()

        delta = np.multiply(delta_input, self.layers[-1].activation.activate_prime(self.get_self_output()))
        dJdW.append(np.dot(dJdW_input.T, delta))
        biases.append(np.sum(delta, axis=0, keepdims=True))

        for num in range(len(self.layers[1:-1]), 0, -1):
            delta_input = np.dot(delta, self.layers[num+1].weights.T) * 2
            dJdW_input = self.layers[num - 1].get_post_output()

            delta = np.multiply(delta_input, self.layers[num].activation.activate_prime(self.layers[num].get_pre_output()))
            dJdW.append(np.dot(dJdW_input.T, delta))
            biases.append(np.sum(delta, axis=0, keepdims=True))
            pass

        delta_input = np.dot(delta, self.layers[1].weights.T) * 2
        dJdW_input = self.inputs

        delta = np.multiply(delta_input, self.layers[0].activation.activate_prime(self.layers[0].get_pre_output()))
        dJdW.append(np.dot(dJdW_input.T, delta))
        biases.append(np.sum(delta, axis=0, keepdims=True))

        self.shell_network(dJdW, biases)
        # for num, dj in enumerate(reversed(dJdW)):
        #     if symbol == '+':
        #         self.layers[num].weights = self.layers[num].weights + (self.scalar*dj)
        #     elif symbol == '-':
        #         self.layers[num].weights = self.layers[num].weights - (self.scalar * dj)
        #     else:
        #         self.error += 1

        return dJdW

    def get_output(self):
        return self.layers[self.current_layer].get_post_output()

    def get_self_output(self):
        return self.layers[self.current_layer].get_pre_output()

    def create_matrix(self, data_outputs=None, classes=None):
        truth_data = self.outputs if data_outputs is None else data_outputs
        no_class_needed = False
        if classes is None:
            classes = self.classes
            no_class_needed = not no_class_needed

        if isinstance(truth_data, np.ndarray):
            return truth_data if no_class_needed else (truth_data, classes)

        elif len(truth_data) > 0 and type(truth_data[0]) is list:
            return (np.array(truth_data)) if no_class_needed else (np.array(truth_data), classes)

        true_matrix = []
        for val in truth_data:
            matrix_row = []
            if type(val) is list: exit()
            for i in range(classes):
                matrix_row.append(1 if i == int(val) else 0)
            true_matrix.append(matrix_row)
        return (np.array(true_matrix)) if no_class_needed else (np.array(true_matrix), classes)

    @staticmethod
    def check_for_matrix(inp):
        return inp if type(inp) is np.ndarray else np.array(inp) if (type(inp[0]) is list or type(inp) is tuple) else np.array([inp])

    def __str__(self):
        return str(self.layers[-1].output)


class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = 0.2 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # might need () around 1, n_neurons
        self.activation = activation
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activation.activate(self.output)

    def backward(self, targets):
        pass

    def get_pre_output(self):
        return self.output

    def get_post_output(self):
        return self.activation.output


class Activation:
    def __init__(self):
        self.output = None

    def activate(self, inputs):
        # sigmoid function
        self.output = 1 / (1 + np.exp(-inputs))

    @staticmethod
    def activate_prime(inputs):
        out = np.exp(-inputs) / ((1 + np.exp(-inputs)) ** 2)
        return np.exp(-inputs) / ((1 + np.exp(-inputs)) ** 2)


class ActivationRelu(Activation):
    def __init__(self):
        super().__init__()

    def activate(self, inputs):
        self.output = np.maximum(0, inputs)
        pass

    @staticmethod
    def activate_prime(inputs):
        a = np.maximum(0, 1, inputs)
        return a


class ActivationOutput(Activation):
    def __init__(self):
        super().__init__()

    def activate(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# def get_data():
#     data = np.array([[3, 5], [5, 1], [8, 3], [8, 6]], dtype=float)
#     data = data / np.amax(data, axis=0)
#     # results = np.array([[75], [82], [93], [95]], dtype=float) / 100
#     results = [75, 82, 93, 95]
#     return data, results


def collect_config():
    import FileOpener

    data = FileOpener.getConfig()

    data['hidden layers'].insert(0, data['inputs'])
    data['hidden layers'].append(data['outputs'])

    return data


def check_accuracy(predictions, actual):
    right = 0
    wrong = 0
    for val1, val2 in zip(predictions, actual):
        if val1 != val2:
            wrong += 1
        else:
            right += 1
    return right, wrong


def run_checks(neural_network, input_data, answers, output_size=None):
    right, wrong = 0, 0
    for input_set, output in zip(input_data, answers):
        prediction = neural_network.predict_input(input_set)
        if prediction[0] == output:
            right += 1
        else:
            wrong += 1
    print("right:", right, "wrong:", wrong)
    return right, wrong


def main():
    global mk_log
    data = collect_config()
    mk_log = data['mk_log']

    X, y = data_management.get_data(data['input file'], data['output file'], data['map file'])
    training_groups_inp, training_groups_out, rest_inp, rest_out = data_management.split_into_groups(data['training sets'], data['training set size'], X, y)

    n = NeuralNetwork(X, data['hidden layers'], y, scalar=data['scalar'])
    for num in range(len(training_groups_inp)):
        n.change_inputs_outputs(training_groups_inp[num], training_groups_out[num])
        n.train_network(data['runs per set'], debug=data['debug'])
        if mk_log:
            logger.write("Next Training Set")

    if mk_log:
        logger.write("No more training sets\nOnto testing accuracy")

    run_checks(n, rest_inp, rest_out)

    n.change_inputs_outputs(rest_inp, [])
    n.run_network()
    right, wrong = check_accuracy(n.get_predictions(), rest_out)

    if mk_log:
        np.set_printoptions(threshold=sys.maxsize)
        logger.write(f"Predictions          -> {('[' + ' '.join([str(val) for val in list(n.get_predictions().tolist())]) + ']')}")
        logger.write(f"Correct Predictions  -> {('[' + ' '.join([str(val) for val in rest_out]) + ']')}")
        logger.write(f"right: {right}, wrong: {wrong}")
        logger.close()

    print(f"The training was {round((right / (right + wrong)) * 100, 2)}% successful")
    print(f"right: {right}, wrong: {wrong}")

    pass


if __name__ == '__main__':
    main()
    pass


"""
    X = Inputs
    y = Outputs
    def test_loss_calc(self):
        yHat = self.get_output()
        output = self.outputs
        multo = self.outputs - yHat
        multo2 = -(self.outputs - yHat)
        non_trans = self.layers[-2].get_post_output()
        trans = self.layers[-2].get_post_output().T
        sig_prime = self.layers[-1].activation.activate_prime(self.get_self_output())
        delta3 = np.multiply(-(self.outputs - yHat), self.layers[-1].activation.activate_prime(self.get_self_output()))
        dJdW2 = np.dot(self.layers[-2].get_post_output().T, delta3)
        pass

"""