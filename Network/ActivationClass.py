import numpy as np
import copy


class Activate:
    def __init__(self, **kwargs):
        self.output = None

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def activate_prime(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    @staticmethod
    def validate():
        return True

    def __str__(self):
        return 'activate'


class Sigmoid(Activate):
    def activate(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def activate_prime(self, inputs):
        a = np.exp(-inputs)
        b = ((1 + np.exp(-inputs)) ** 2)
        return np.exp(-inputs) / ((1 + np.exp(-inputs)) ** 2)

    def __str__(self):
        return 'sigmoid'


class Relu(Activate):

    def activate(self, inp):
        self.output = np.maximum(0, inp)
        return self.output

    def activate_prime(self, inputs):
        inputs = getCopy(inputs)
        inputs[inputs <= 0] = 0
        inputs[inputs > 0] = 1
        return inputs

    def __str__(self):
        return 'relu'


class ReluLeaky(Activate):
    def __init__(self, **kwargs):
        super().__init__()
        self.leaky = kwargs.get('leak', 1)

    def activate(self, inp):
        self.output = getCopy(inp)
        self.output[self.output < 0] /= self.leaky
        val = np.max(np.abs(self.output), axis=0, keepdims=True)
        val[val == 0] = 1e-10
        self.output = self.output / val
        return self.output

    def activate_prime(self, inputs):
        inputs = getCopy(inputs)
        inputs[inputs < 0] = 1 / self.leaky
        inputs[inputs == 0] = 0
        inputs[inputs > 0] = 1
        return inputs

    def __str__(self):
        return 'leakyrelu'


class SoftMax(Activate):

    def __init__(self, **kwargs):
        super().__init__()
        self.activate_prime_func = kwargs.get('prime_activate', None)

    def activate(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def activate_prime(self, inp):
        if self.activate_prime_func:
            return self.activate_prime_func.activate_prime(inp)
        return inp

    def __str__(self):
        return 'softmax'


def getCopy(inp):
    return copy.deepcopy(inp)
