import numpy as np
from utils import linear, relu, sigmoid, softmax


class FFNNLayer:
    def __init__(self, number_of_neurons: int, activation_function: str):
        """
        :param number_of_neurons:
        :param activation_function:
        """
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class FFNN:
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
        self.prediction = None
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
            if act_func == 'linear':
                res = [linear(x) for x in net]
            if act_func == 'relu':
                res = [relu(n) for n in net]
            if act_func == 'sigmoid':
                res = [sigmoid(n) for n in net]
            if act_func =="softmax":
                res = [softmax(n) for n in net]
        self.prediction = res
        return res

    def calculate_sse(self):
        """
        calculate Sum Squared Error (SSE) of result and expected

        :return: sum squared error
        """
        expected = np.array(self.Y_expected[0])
        squared_error = (expected - self.prediction[0]) ** 2
        sum_squared_error = np.sum(squared_error)
        return sum_squared_error
    
    def print_expected_output(self):
        """
        prints the expected output in a formatted way.

        :return: void
        """
        print(f"Expected Output:")
        for i, sublist in enumerate(self.Y_expected):
            for j, val in enumerate(sublist):
                print(f"Output {i+1}.{j+1}: {val:.4f}")
        print("-" * 20)  # Separator 

    def print_prediction_results(self):
        """
        prints the prediction results in a formatted way.

        :return: void
        """
        print(f"Prediction Result:")
        np.set_printoptions(precision=4, suppress=True)
        print(self.Y)
        for i, result in enumerate(self.prediction):
            print(f"Input {i+1}:")
            if isinstance(result, np.ndarray):
                for j, val in enumerate(result.flatten()): 
                    print(f"  Output {i+1}.{j+1}: {val:.4f}")
            else:
                print(f"  Output: {result:.4f}")
            print("-" * 20)  # Separator 
        