import math
import numpy as np
from utils import linear_net_gradient, relu_net_gradient, sigmoid, relu, softmax, linear, sigmoid_net_gradient, softmax_net_gradient


class FFNNLayer:
    def __init__(self, number_of_neurons: int, activation_function: str):
        """
        :param number_of_neurons:
        :param activation_function:
        """
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class MLPClassifier:
    def __init__(self, layers: list, learning_rate, error_threshold, max_iter, batch_size, weights):
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
        self.weights = [weight[1:] for weight in weights]
        self.bias_weights = [weight[0] for weight in weights]
        self.prediction = []
        self.num_of_features = 0
        self.num_of_batches = 0
        self.d_weights = None
        self.d_bias_weights = None
        self.stopped_by = None


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_features = len(self.X_train)
        self.num_of_batches = math.ceil(len(self.X_train) / self.batch_size)

        # the first neuron is the X inputs themselves
        self.neuron_values = [[None for _ in range(layer.number_of_neurons)] for layer in self.layers]
        num_iter = 0
        while self.error_sum > self.error_threshold and num_iter < self.max_iter:
            num_of_batches = math.ceil(len(self.X_train) / self.batch_size)
            for i in range(num_of_batches):
                self.__forward(i)
                self.__backward(i)
                # todo: update error (@chow)
                # jujur ga tau harusnya di bagian mana wkakwoka

            num_iter += 1
        
        self.stopped_by = "max_iteration" if num_iter == self.max_iter else "error_threshold"

        print(self.weights)
        print(self.bias_weights)

    def predict(self, X_test):
        """Perform forward pass to make predictions on input X_test

        Args:
            X_test: Input data for prediction (list)

        Returns:
            Predicted outputs for each sample in X_test
        """
        predictions = []
        current_inputs = np.array(X_test) 
        for i in range(self.num_of_layers):
            net = np.matmul(current_inputs, self.weights[i]) + self.bias_weights[i]
            act_func = self.layers[i].activation_function
            if act_func == 'linear':
                res = [linear(x) for x in net]
            elif act_func == 'relu':
                res = [relu(n) for n in net]
            elif act_func == 'sigmoid':
                res = [sigmoid(n) for n in net]
            elif act_func == "softmax":
                res = [softmax(n) for n in net]
            current_inputs = res
        predictions = res.toList()
        return predictions
    
    def calculate_sse(self, final_weights):
        sse = 0
        for layer in range(len(final_weights)):
            for neuron in range(len(final_weights[layer])):
                expected = np.array(final_weights[layer][neuron])
                result = self.bias_weights[layer] if neuron == 0 else self.weights[layer][neuron]
                squared_error = (expected - result) ** 2
                sse += np.sum(squared_error)
        return sse


    def __forward(self, batch):
        start_idx = self.batch_size * batch
        res = self.X_train[start_idx:start_idx + self.__get_curr_batch_size(batch)]
        for i in range(self.num_of_layers):
            net = [np.matmul(x, self.weights[i]) + self.bias_weights[i] for x in res]
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
        self.prediction = list(self.neuron_values[-1])
        return self.prediction

    def __backward(self, batch_idx):
        """
        do backward propagation for each batch
        :param batch_idx: the current batch that is processed
        """
        self.__init_d_weights()
        # get the current batch size
        batch_size = self.__get_curr_batch_size(batch_idx)



        # for each X in the batch
        for i in range(batch_size):
            x_idx = batch_idx * batch_size + i  # x_idx is the index of the current input on the X_train
            d_k = np.zeros(0)
            for j in range(self.num_of_layers - 1, -1, -1):
                if j == self.num_of_layers - 1:
                    delta = self.__calc_output_layer_delta(x_idx)
                else:
                    delta = self.__calc_hidden_layer_delta(i, j, d_k)

                x = self.X_train[x_idx] if j == 0 else self.neuron_values[j - 1][i]
                self.d_weights[j] += np.array([[d * n for d in delta] for n in x])
                self.d_bias_weights[j] += np.array(delta)
                d_k = delta.reshape(delta.shape[0], 1)

        self.weights = [np.array(self.weights[k]) + np.array(self.d_weights[k]) * self.learning_rate for k in
                        range(len(self.weights))]
        self.bias_weights = [np.array(self.bias_weights[k]) + np.array(self.d_bias_weights[k]) * self.learning_rate for
                             k in range(len(self.bias_weights))]
        # todo: (@chow)
        # calc error
        # kayaknya somewhere disini

    def __init_d_weights(self):
        self.d_weights = [np.array([np.zeros(len(neuron_weight)) for neuron_weight in layer_weight])
                          for layer_weight in self.weights]
        self.d_bias_weights = [np.zeros(layer.number_of_neurons) for layer in self.layers]

    def __calc_output_diff(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        y_train = self.y_train[x_idx]  # get the expected output of the x
        output = self.prediction[x_idx]  # get the prediction
        return np.array([y - p for y, p in zip(y_train, output)])

    def __calc_act_function_derivative(self, act_func: str, y: list, target) -> np.ndarray:
        """
        :param y:  y is the output in a layer

        :return : a 1D array which is the sigmoid gradient of the neurons in a layer
        """
        if act_func == 'sigmoid':
            return np.array([x * (1-x) for x in y])

        elif act_func == 'relu':
            return np.array([1 if x >= 0 else 0 for x in y])

        elif act_func == 'linear':
            return np.arrat([1 if x > 0 else 0 for x in y])

        elif act_func == 'softmax':
            if target is None:
                raise ValueError("Target is required for softmax gradient")
            grad = np.copy(y)
            grad[target] = grad[target] - 1
            return grad

        else:
            raise ValueError(f"Unknown activation function: {act_func}")


    def __calc_output_layer_delta(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        # get the activation function for the last layer (output layer)
        act_func = self.layers[-1].activation_function  # get the activation function
        return self.__calc_act_function_derivative(act_func, self.prediction[x_idx]) * self.__calc_output_diff(x_idx)

    def __calc_hidden_layer_delta(self, batch_idx, layer_idx: int, output_error_term: np.ndarray) -> np.ndarray:
        """
        :param output_error_term: a 1D array of the error term of each weight calculated from the layer after
        :param layer_idx: the index of the current layer
        :param batch_idx: the index of the current batch

        hidden layer gradient = net gradient of the neuron values of current layer * the sum of weight * output error term
        """
        act_func = self.layers[layer_idx].activation_function
        activation_func_derivative = self.__calc_act_function_derivative(act_func,
                                                                         self.neuron_values[layer_idx][batch_idx])
        return np.array(activation_func_derivative
                        * np.matmul(self.weights[layer_idx + 1], output_error_term)[0])

    def __get_curr_batch_size(self, batch_idx):
        mod_res = len(self.X_train) % self.batch_size
        if batch_idx == self.batch_size - 1 and mod_res != 0:
            return mod_res
        return self.batch_size
