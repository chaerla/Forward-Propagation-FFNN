import numpy as np
from utils import relu, sigmoid

class FFNNLayer():
    def __init__(self, number_of_neurons: int, activation_function: str):
        """
        :param number_of_neurons:
        :param activation_function:
        """
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class FFNN():
    def __init__(self, input_size: int, layers: list, weights: list):
        """
        initializes the model

        :param input_size: the size of input to be received
        :param layers: list of FFNNLayer that specifies each layer's neuron count and activation function
        :param weights: list of weight on each layer
        """
        self.input_size = input_size
        self.number_of_layers = len(layers)
        self.layers = layers
        self.X = []
        self.Y = []
        self.Y_expected = None
        self.weights = [np.array(l) for l in weights]

    def fit(self, x:list, y_expected=None):
        """
        fit the test x and y_expected to the model

        :param x: X inputs to be predicted
        :param y_expected: expected y
        :return: void
        """
        if y_expected is None:
            y_expected = []
        self.X = x
        self.Y_expected = y_expected

    def predict(self):
        """
        calculates the output by doing forward propagation

        :return: output of each X
        """
        res = self.X
        for i in range(self.number_of_layers):
            res = [np.insert(x, 0, 1) for x in res]
            net = [np.matmul(x, self.weights[i]) for x in res]
            act_func = self.layers[i].activation_function
            if act_func == 'relu':
                res = [relu(n) for n in net]
            if act_func == 'sigmoid':
                res = [sigmoid(n) for n in net]
        return res




