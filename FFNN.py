import numpy as np
from utils import relu, sigmoid

# use this to read from JSON and convert to this class
class FFNNLayer():
    def __init__(self, number_of_neurons: int, activation_function: str):
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class FFNN():
    """
    initialize the model,
    input_size: int is the size of input to be received
    layers:list of FFNNLayer that specifies each layer's neuron count and activation function
    weights: list of weight on each layer
    """
    def __init__(self, input_size: int, layers: list, weights: list):
        self.input_size = input_size
        self.number_of_layers = len(layers)
        self.layers = layers
        self.X = []
        self.Y = []
        self.Y_expected = None
        self.weights = [np.array(l) for l in weights]

    """
    fit the test x and y_expected to the model
    """
    def fit(self, x:list, y_expected=None):
        if y_expected is None:
            y_expected = []
        for el in x:
            el.insert(0, 1)
            self.X.append(np.array(el))
        self.Y_expected = y_expected

    """
    calculates the output by doing forward propagation
    
    return output of each X
    """
    def predict(self):
        res = self.X
        for i in range(self.number_of_layers):
            net = [np.matmul(x, self.weights[i]) for x in res]
            act_func = self.layers[i].activation_function
            if act_func == 'relu':
                res = [np.insert(relu(n), 0, 1) for n in net]
            if act_func == 'sigmoid':
                res = [np.insert(sigmoid(n), 0, 1) for n in net]
        return [np.delete(r, 0) for r in res]




