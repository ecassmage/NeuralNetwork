from Network import ActivationClass, Layer
# import copy
import numpy as np


class NeuralNetwork:
    def __init__(self, **kwargs):
        """
        kwargs: {
            config: mandatory,
            presets: mandatory,
            activation: not mandatory,
            inp: not mandatory,
            out: not mandatory,
            h_layers: not mandatory,
            h_layers_range: not mandatory,
            scalar: not mandatory,
            log: not mandatory,
        }
        :param kwargs:
        """
        self.activation_name = 'sigmoid'
        self.activation_default = self.get_activation(self.activation_name)
        self.h_layers_range = (-1, 1)
        self.leak = 1

        self.softmax = False

        self.debug = False  # debug
        self.log = None  # log

        self.classes = 0
        self.input = None
        self.output = None

        self.hidden_layers = []
        self.layers = []

        self.scalar = 1

        self.set_network(**kwargs)
        pass

    def think(self, inp): return self.__think(inp)
    def train(self, number_of_iterations=1): return self.__train(number_of_iterations)
    def get_chosen_outputs(self): return self.__get_chosen_outputs()

    def set_network(self, **kwargs):
        kwargs = set_config(kwargs.get('config', {}), kwargs.get('preset', {}), kwargs)  # config merging

        self.activation_name = kwargs.pop('activation', self.activation_name)
        self.activation_default = self.get_activation(self.activation_name)

        self.leak = kwargs.pop('leak', self.leak)
        self.h_layers_range = kwargs.pop('h_layers_range', self.h_layers_range)
        self.softmax = kwargs.pop('softmax_output', self.softmax)

        self.debug = kwargs.get('debug', self.debug)  # debug
        self.log = kwargs.get('log', self.log)  # log

        self.classes = kwargs.get('classes', self.classes)
        self.input = check_matrix(kwargs.get('input', self.input))
        self.output = check_matrix(kwargs.get('output', self.output), self.classes)

        self.hidden_layers = self.set_h_layers(**kwargs, hidden_layers=self.hidden_layers)

        self.layers = self.set_layers(**kwargs, layers=self.layers)

        self.scalar = kwargs.get('scalar', self.scalar)
        pass

    def __think(self, inp):
        """
        Takes a set of inputs, puts them through the network and returns what it thinks is the valid answer.

        :param inp:
        :return:
        """
        for layer in self.layers:
            layer.forward(inp)
            inp = layer.get_output()
        return self.get_chosen_outputs()

    def __train(self, number_of_iterations):
        """
        Trains the network.

        :param number_of_iterations:
        :return:
        """

        for _ in range(number_of_iterations):
            self.think(self.input)
            if self.log: self.log.write_loss(self)
            self.loss()
        pass

    @staticmethod
    def get_activation(act):
        a = str(act)
        match str(act).lower():
            case 'sigmoid':
                return ActivationClass.Sigmoid
            case 'relu':
                return ActivationClass.Relu
            case 'leakyrelu':
                return ActivationClass.ReluLeaky
            case _:
                pass

    def copy(self, number_of_copies=1):
        return [
            NeuralNetwork(
                inp=list(self.input.tolist()),
                out=list(self.output.tolist()),
                classes=self.classes,
                h_layers=self.hidden_layers[:-1],
                scalar=self.scalar,
                activation=self.activation_name,
                softmax_output=self.softmax,
                leak=self.leak,
                h_layers_range=self.h_layers_range

            ) for _ in range(number_of_copies)
        ]

    @staticmethod
    def set_h_layers(**kwargs):
        temp = ([kwargs['inputs']] if 'inputs' in kwargs else []) + kwargs.get('h_layers', []) + ([kwargs['classes']] if 'classes' in kwargs else [])
        return temp if len(temp) > 0 else kwargs.get('hidden_layers', [])

    def set_layers(self, **kwargs):
        """
        Needs to add feature for softmax
        Also needs to work
        :param kwargs:
        :return:
        """
        kwargs.update({'leak': self.leak, 'softmax': self.softmax, 'h_layers_range': self.h_layers_range})
        layers = [Layer.Layer(n_inputs=self.hidden_layers[num], n_neurons=val, activation=self.activation_default(**kwargs), **kwargs) for num, val in enumerate(self.hidden_layers[1:])] if len(self.hidden_layers) > 1 else kwargs.get('layers', [])

        if len(layers) > 0 and self.softmax:
            layers[-1].activation = ActivationClass.SoftMax(prime_activate=layers[-1].activation)

        return layers

    def CHECK_NETWORK(self):
        if not self.input or not self.output or not self.layers:
            return False

        previous = len(self.input[0])
        for layer in self.layers:
            weight_shape = layer.weights.shape
            bias_shape = layer.biases.shape

            if weight_shape[0] != previous or bias_shape[1] != weight_shape[1] or layer.activation.validate():  # layer.activation.validate() will crash if it fails because it would mean it doesn't exist.
                return False

            previous = weight_shape[1]

        return previous != len(self.output[0]) or len(self.input) != len(self.output)

    def shell_network(self, new_weights, biases):
        """
        Builds a set of new neural networks and tests to see which one is best.
        Might need to remake this one.

        :param new_weights:
        :param biases:
        :return:
        """

        change_in_cost = []
        neural_cost = self.get_sum_of_losses()

        network = self.copy()[0]
        for ternary_set in range(8):
            weight_bit, bias_bit = convert_base(ternary_set, 3, bits=2)
            for num, (weight, bias) in enumerate(reversed(tuple(zip(new_weights, biases)))):
                match weight_bit:
                    case 0:
                        network.layers[num].set_weights(self.layers[num].weights + (self.scalar * weight))
                    case 1:
                        network.layers[num].set_weights(self.layers[num].weights - (self.scalar * weight))
                    case 2:
                        pass
                match bias_bit:
                    case 0:
                        network.layers[num].set_biases(self.layers[num].biases + (self.scalar * bias))
                    case 1:
                        network.layers[num].set_biases(self.layers[num].biases - (self.scalar * bias))
                    case 2:
                        pass
            network.think(self.input)
            change_in_cost.append(neural_cost - network.get_sum_of_losses())

        index = change_in_cost.index(max(change_in_cost))
        for num, (weight, bias) in enumerate(reversed(tuple(zip(new_weights, biases)))):
            match index:
                case 0:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * weight)
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * bias)
                    pass
                case 1:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * weight)
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * bias)
                    pass
                case 2:
                    self.layers[num].weights = self.layers[num].weights + (self.scalar * weight)
                    pass
                case 3:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * weight)
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * bias)
                    pass
                case 4:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * weight)
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * bias)
                    pass
                case 5:
                    self.layers[num].weights = self.layers[num].weights - (self.scalar * weight)
                    pass
                case 6:
                    self.layers[num].biases = self.layers[num].biases + (self.scalar * bias)
                    pass
                case 7:
                    self.layers[num].biases = self.layers[num].biases - (self.scalar * bias)
                    pass
                case _:
                    exit()

    def loss(self):
        """
        Calculates the loss of the neural network.

        :return:
        """

        dJdW = []
        biases = []

        def append(new_dJdW, new_bias):
            dJdW.append(new_dJdW)
            biases.append(new_bias)

        delta_input = -(self.output - self.get_output(activated=True)) * 2
        dJdW_input = self.layers[-2].get_output(activated=True)

        a = delta_input
        aa = self.layers[-1].activation.activate_prime(self.get_output(activated=False))

        delta = np.multiply(delta_input, self.layers[-1].activation.activate_prime(self.get_output(activated=False)))
        append(np.dot(dJdW_input.T, delta), np.sum(delta, axis=0, keepdims=True))

        for num in range(len(self.layers[1:-1]), 0, -1):

            delta_input = np.dot(delta, self.layers[num + 1].weights.T) * 2
            dJdW_input = self.layers[num - 1].get_output(activated=True)

            delta = np.multiply(delta_input, self.layers[num].activation.activate_prime(self.layers[num].get_output(activated=False)))
            append(np.dot(dJdW_input.T, delta), np.sum(delta, axis=0, keepdims=True))

        delta_input = np.dot(delta, self.layers[1].weights.T) * 2
        dJdW_input = self.input

        delta = np.multiply(delta_input, self.layers[0].activation.activate_prime(self.layers[0].get_output(activated=False)))
        append(np.dot(dJdW_input.T, delta), np.sum(delta, axis=0, keepdims=True))

        self.shell_network(dJdW, biases)

    def get_output(self, activated=True):
        return self.layers[-1].get_output(activated)

    def get_sum_of_losses(self):

        a = self.get_output()
        b = self.output
        c = (a - b)
        d = c ** 2
        e = np.sum(d)
        return np.sum((self.get_output() - self.output) ** 2)

    def __get_chosen_outputs(self):
        return np.argmax(self.get_output(), axis=1)


def get_higher_set(name, value, **kwargs):
    temp = kwargs.get(name, value)
    return temp if temp is not None else list()


def dimensionality(arr):
    if isinstance(arr, list) or isinstance(arr, tuple) or isinstance(arr, np.ndarray):
        return 1 + (dimensionality(arr[0]) if len(arr) > 0 else 0)
    return 0


def set_config(config_file, preset, kwargs):
    def normalize_key():
        key_norm = key.lower().replace(' ', '_')
        match key_norm:
            case 'inp':
                return 'input'
            case 'out':
                return 'output'
            case 'hidden_layers' | 'hidden_layer':
                return 'h_layers'
            case 'logger':
                return 'log'
            case 'debugger':
                return 'debug'
            case 'activator':
                return 'activation'
            case _:
                return key_norm

    config = {}

    for key in config_file:  # lowest precedence (defaults)
        if key == 'presets' or key == 'preset':
            continue
        config[normalize_key()] = config_file[key]

    for key in preset:  # second precedence
        config[normalize_key()] = preset[key]

    for key in kwargs:  # highest precedence
        if key in ('presets', 'preset', 'config', 'configs'):
            continue
        config[normalize_key()] = kwargs[key]

    return config


def check_matrix(array, classes=0):
    if isinstance(array, np.ndarray):
        return array, classes
    if not array:
        return None
    match dimensionality(array):
        case 1:
            for num, val in enumerate(array):
                array[num] = [0] * classes
                array[num][int(val)] = 1
            return np.array(array)
        case 2:
            return np.array(array)
        case _:
            exit()
    pass


def convert_base(integer, base, bits=0):
    arr = []
    while not integer < base:
        number = float(integer) / base
        arr.insert(0, round((number - int(number)) * base))
        integer = int(number)

    arr.insert(0, round(integer))
    return [0 for _ in range(bits - len(arr))] + arr if bits else arr


def test():
    print(dimensionality('a'))  # 0
    print(dimensionality(['a']))  # 1
    print(dimensionality(('a',)))  # 1
    print(dimensionality(np.array([[['a']]])))  # 3
    print(dimensionality([['a']]))  # 2
    print(dimensionality([[]]))  # 2
    print(dimensionality([]))  # 1
    print(dimensionality([('a',)]))  # 2
    print(dimensionality(np.array(['a'])))  # 1


if __name__ == '__main__':
    test()
    pass
