import math
import numpy as np
from utils import sigmoid, relu, softmax, linear, sigmoid_net_gradient


class FFNNLayer:
    def __init__(self, number_of_neurons: int, activation_function: str):
        """
        :param number_of_neurons:
        :param activation_function:
        """
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class MLPClassifier:
    def __init__(self, layers: list, learning_rate, error_threshold, max_iter, batch_size):
        """
        :param layers: list of FFNNLayer to specify the activation function and num of neurons for each layers
        :param learning_rate: the learning rate
        :param error_threshold: the error threshold
        :param max_iter: max iter to stop iteration
        :param batch_size: the size of batch for each mini batch
        """
        self.num_of_layers = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self.error_sum = 1
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.X_train = []
        self.y_train = []
        self.neuron_values = []
        self.weights = []
        self.prediction = []
        self.num_of_features = 0
        self.num_of_batches = 0

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_features = len(self.X_train)
        self.num_of_batches = math.ceil(len(self.X_train) / self.batch_size)

        # the first neuron is the X inputs themselves
        self.neuron_values = [[None for j in range(layer.number_of_neurons)] for layer in self.layers]

        # prep the initial weights
        for i in range(self.num_of_layers):
            num_of_source_neurons = len(self.X_train[0]) if i == 0 else self.layers[i - 1].number_of_neurons
            num_of_target_neurons = self.layers[i].number_of_neurons
            self.weights.append(self.__initialize_layer_weight(num_of_source_neurons, num_of_target_neurons))

    def predict(self):
        if self.X_train is None or self.y_train is None:
            raise Exception("Model is not fitted yet")
        num_iter = 0
        while self.error_sum > self.error_threshold and num_iter < self.max_iter:
            num_of_batches = math.ceil(len(self.X_train) / self.batch_size)
            for i in range(num_of_batches):
                self.__forward(i)
                self.__backward(i)
                # todo: update error (@chow)
            num_iter += 1

    def __forward(self, batch_num):
        res = self.X_train
        for i in range(self.num_of_layers):
            res = [np.insert(x, 0, 1) for x in res]
            net = [np.matmul(x, self.weights[i]) for x in res]
            act_func = self.layers[i].activation_function
            if act_func == 'linear':
                res = [linear(x) for x in net]
            if act_func == 'relu':
                res = [relu(n) for n in net]
            if act_func == 'sigmoid':
                res = [sigmoid(n) for n in net]
            if act_func == "softmax":
                res = [softmax(n) for n in net]
            self.neuron_values[i] = res
        self.prediction += list(self.neuron_values[self.num_of_layers - 1])
        return self.prediction

    def __backward(self, batch_idx):
        """
        do backward propagation for each batch
        :param batch_idx: the current batch that is processed
        """
        d_bias = []
        # get the current batch size
        batch_size = self.__get_curr_batch_size(batch_idx)

        # initialize a 2D numpy array, each row has the gradient for each neuron
        # e.g.
        # neuron        0  1   2
        # [
        # layer 1      [0, 0,  0],
        # layer 2      [0, 0,  0],
        # ]
        # each element represents the sum of the gradient of the neuron from each x in the batch
        gradient_sum = [np.zeros(self.layers[i].number_of_neurons, dtype=float) for i in range(self.num_of_layers)]

        # todo: (@livia)
        # save the bias sum to update bias as well

        # todo: (@chow)
        # calc error

        # for each X in the batch
        for i in range(batch_size):
            x_idx = batch_idx * batch_size + i  # x_idx is the index of the current input on the X_train
            output_error_term = 0
            # calc the gradient on each layer
            # gradient is a 1D numpy array containing gradient for each neuron in the layer
            for j in range(self.num_of_layers - 1, -1, -1):
                if j == self.num_of_layers - 1:
                    gradient = self.__calc_output_gradient(x_idx)
                else:
                    gradient = self.__calc_hidden_layer_gradient(i, j, output_error_term)
                gradient_sum[j] += gradient
                output_error_term = gradient

        # the mean gradient is the gradient sum divided by the batch size
        mean_gradient = [gradient / batch_size for gradient in gradient_sum]
        # calc the mean bias change(?)

        # todo:
        # update the weight (@livia)
        # update the bias (@livia)


    def __calc_output_diff(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        y_train = self.y_train[x_idx]  # get the expected output of the x
        output = self.prediction[x_idx]  # get the prediction
        return np.array([y - p for y, p in zip(y_train, output)])

    def __calc_net_gradient(self, act_func: str, y: list) -> np.ndarray:
        """
        :param y:  y is the output in a layer

        :return : a 1D array which is the sigmoid gradient of the neurons in a layer
        """
        if act_func == 'sigmoid':
            return np.array(sigmoid_net_gradient(y))

        # todo: handle other functions (@jason)

    def __calc_output_gradient(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        # get the activation function for the last layer (output layer)
        act_func = self.layers[self.num_of_layers - 1].activation_function  # get the activation function
        return self.__calc_net_gradient(act_func, self.prediction[x_idx]) * (self.__calc_output_diff(x_idx))

    def __calc_hidden_layer_gradient(self, batch_idx, layer_idx: int, output_error_term: np.ndarray) -> np.ndarray:
        """
        :param output_error_term: a 1D array of the error term of each weight calculated from the layer after
        :param layer_idx: the index of the current layer
        :param batch_idx: the index of the current batch

        hidden layer gradient = net gradient of the neuron values of current layer * the sum of weight * output error term
        """
        act_func = self.layers[layer_idx].activation_function
        return (self.__calc_net_gradient(act_func, self.neuron_values[batch_idx][layer_idx]) *
                (np.sum(np.transpose(self.weights[layer_idx]) * output_error_term)))

    def __get_curr_batch_size(self, batch_idx):
        mod_res = len(self.X_train) % self.batch_size
        if batch_idx == self.batch_size - 1 and mod_res != 0:
            return mod_res
        return self.batch_size

    def __initialize_layer_weight(self, num_of_source_neurons, num_of_target_neurons):
        """
        this function generates the weight for each layer

        :param num_of_source_neurons: the number of neurons in the source layer
        :param num_of_target_neurons: the number of neurons in the target layer

        :return : a matrix of size num_of_source_neurons + 1 (1 for bias) * num_of_target neurons
        with random values between -0.5 to 0.5
        """
        return np.random.rand(num_of_source_neurons + 1, num_of_target_neurons) - 0.5
