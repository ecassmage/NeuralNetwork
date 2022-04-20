import numpy as np


class Layer:
    def __init__(self, **kwargs):
        # self.config = {}
        self.weights = None
        self.biases = None
        self.activation = None
        self.output = None

        self.load_layer(kwargs['weights'], kwargs['biases'], kwargs['activation']) \
            if 'weights' in kwargs else \
            self.random_layer(kwargs['n_inputs'], kwargs['n_neurons'], kwargs['activation'], layer_range=kwargs.get('h_layers_range', (-1, 1)))

    def load_layer(self, weights, biases, activation):
        self.set_weights(weights)
        self.set_biases(biases)
        self.activation = activation

    def random_layer(self, number_of_inputs, number_of_neurons, activation, layer_range=(-1, 1)):
        self.set_weights(np.random.uniform(layer_range[0], layer_range[1], (number_of_inputs, number_of_neurons)))
        self.set_biases(np.zeros((1, number_of_neurons)))
        self.activation = activation

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.activate()
        # self.activation.activate(self.output)

    def activate(self, prime=False):
        return self.activation.activate_prime(self.output) if prime else self.activation.activate(self.output)

    def get_output(self, activated=True):
        return self.activation.output if activated else self.output

    def set_weights(self, new_weights):
        self.weights = convert(new_weights, np.ndarray, np.array)

    def set_biases(self, new_biases):
        self.biases = convert(new_biases, np.ndarray, np.array)


def convert(obj, d_type, convert_func):
    return obj if isinstance(obj, d_type) else convert_func(obj)
