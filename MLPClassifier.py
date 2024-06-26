import math
import numpy as np
import os
import json
from utils import sigmoid, relu, softmax, linear


class FFNNLayer:
    def __init__(self, number_of_neurons: int, activation_function: str):
        """
        :param number_of_neurons:
        :param activation_function:
        """
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function


class MLPClassifier:
    def __init__(self, layers: list, weights, learning_rate=None, error_threshold=None, max_iter=None, batch_size=None,  stopped_by=None, expected_weights = None):
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
        self.expected_stopped_by = stopped_by
        self.expected_weights = expected_weights
        self.expected_output = None
        self.stopped_by = None
        self.current_inputs = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_features = len(self.X_train)
        self.num_of_batches = math.ceil(len(self.X_train) / self.batch_size)

        # the first neuron is the X inputs themselves
        self.neuron_values = [[None for _ in range(layer.number_of_neurons)] for layer in self.layers]
        num_iter = 0
        while num_iter < self.max_iter:
            num_of_batches = math.ceil(len(self.X_train) / self.batch_size)
            err = 0
            for i in range(num_of_batches):
                self.__forward(i)
                self.__backward(i)
                err += self.__calculate_error(i)

            # Update the average error for this iteration
            self.error_sum = err / num_of_batches

            # Check if the error is below the threshold
            if self.error_sum <= self.error_threshold:
                break

            num_iter += 1

        self.stopped_by = "max_iteration" if num_iter == self.max_iter else "error_threshold"

        if self.expected_weights:
            self.__print_final_weights()

    def predict(self, X_test):
        """Perform forward pass to make predictions on input X_test

        Args:
            X_test: Input data for prediction (list)

        Returns:
            Predicted outputs for each sample in X_test
        """
        current_inputs = np.array(X_test)
        for i in range(self.num_of_layers):
            net = [np.matmul(x, self.weights[i]) + self.bias_weights[i] for x in current_inputs]
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
        return res

    def calculate_sse(self):
        sse = 0
        for layer in range(len(self.expected_weights)):
            for neuron in range(len(self.expected_weights[layer])):
                expected = np.array(self.expected_weights[layer][neuron])
                result = self.bias_weights[layer] if neuron == 0 else self.weights[layer][neuron-1]
                squared_error = (expected - result) ** 2
                sse += np.sum(squared_error)
        return sse

    def __forward(self, batch):
        start_idx = self.batch_size * batch
        self.expected_output = self.y_train[start_idx:start_idx + self.__get_curr_batch_size(batch)]
        self.current_inputs = self.X_train[start_idx:start_idx + self.__get_curr_batch_size(batch)]
        res = self.current_inputs
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
        # print("pred", self.neuron_values[-1])    
        self.prediction = list(self.neuron_values[-1])

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
            d_k = np.zeros(0)
            for j in range(self.num_of_layers - 1, -1, -1):
                if j == self.num_of_layers - 1:       
                    delta = self.__calc_output_layer_delta(i)
                else:
                    delta = self.__calc_hidden_layer_delta(i, j, d_k)
                x = self.current_inputs[i] if j == 0 else self.neuron_values[j - 1][i]
                self.d_weights[j] += np.array([[d * n for d in delta] for n in x])
                self.d_bias_weights[j] += np.array(delta)
                d_k = delta.reshape(delta.shape[0], 1)
        
        self.weights = [np.array(self.weights[k]) + np.array(self.d_weights[k]) * self.learning_rate for k in
                        range(len(self.weights))]
        self.bias_weights = [np.array(self.bias_weights[k]) + np.array(self.d_bias_weights[k]) * self.learning_rate for
                             k in range(len(self.bias_weights))]

    def __calculate_error(self, batch_idx):
        """
        Calculate the error for the current batch
        :param batch_idx: the current batch that is processed
        """
        start_idx = self.batch_size * batch_idx
        end_idx = start_idx + self.__get_curr_batch_size(batch_idx)
        y_true = np.array(self.y_train[start_idx:end_idx])
        y_pred = np.array(self.prediction)

        # Get the activation function of the output layer
        act_func = self.layers[-1].activation_function

        # Calculate the error based on the activation function
        if act_func in ['relu', 'sigmoid', 'linear']:
            return 0.5 * np.sum((y_true - y_pred) ** 2)
        elif act_func == 'softmax':
            return -np.sum(y_true * np.log(y_pred))
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

    def __update_weights(self):
        self.weights = [np.array(self.weights[k]) + np.array(self.d_weights[k]) * self.learning_rate for k in
                        range(len(self.weights))]
        self.bias_weights = [np.array(self.bias_weights[k]) + np.array(self.d_bias_weights[k]) * self.learning_rate for
                             k in range(len(self.bias_weights))]

    def __init_d_weights(self):
        self.d_weights = [np.array([np.zeros(len(neuron_weight)) for neuron_weight in layer_weight])
                          for layer_weight in self.weights]
        self.d_bias_weights = [np.zeros(layer.number_of_neurons) for layer in self.layers]

    def __calc_output_diff(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        y_train = self.expected_output[x_idx]  # get the expected output of the x
        output = self.prediction[x_idx]  # get the prediction
        return np.array([y - p for y, p in zip(y_train, output)])

    def __calc_act_function_derivative(self, act_func: str, y: list, target=None) -> np.ndarray:
        """
        :param y:  y is the output in a layer

        :return : a 1D array which is the sigmoid gradient of the neurons in a layer
        """
        if act_func == 'sigmoid':
            return np.array([x * (1-x) for x in y])

        elif act_func == 'relu':
            return np.array([1 if x > 0 else 0 for x in y])

        elif act_func == 'linear':
            return np.array([1 for _ in y])

        elif act_func == 'softmax':
            if target is None:
                raise ValueError("Target is required for softmax gradient")
            return np.array([-1 * (1-y[i]) if target == i else y[i] for i in range(len(y))])

        else:
            raise ValueError(f"Unknown activation function: {act_func}")


    def __calc_output_layer_delta(self, x_idx: int) -> np.ndarray:
        """
        :param x_idx:  the index of the current input on the X_train
        """
        # get the activation function for the last layer (output layer)
        act_func = self.layers[-1].activation_function  # get the activation function
    
        if act_func == 'softmax':
            return self.__calc_output_diff(x_idx)
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

        sum_d_net = [x[0] for x in np.matmul(self.weights[layer_idx + 1], output_error_term)]
        return np.array(activation_func_derivative
                        * sum_d_net)

    def __get_curr_batch_size(self, batch_idx):
        mod_res = len(self.X_train) % self.batch_size
        if batch_idx == self.num_of_batches - 1 and mod_res != 0:
            return mod_res
        return self.batch_size

    def __print_final_weights(self):
        print("========= EXPECTED =========")
        for weight in self.expected_weights:
            print("[")
            for neuron_weight in weight:
                print("  ", neuron_weight)
            print("], ")
        print("STOPPED BY: ", self.expected_stopped_by)

        print("========== ACTUAL ==========")

        for i in range(len(self.weights)):
            print("[")
            print("  ", self.bias_weights[i])
            for neuron_weight in self.weights[i]:
                print("  ", neuron_weight)
            print("], ")
        print("STOPPED BY: ", self.stopped_by)
    
    def calc_score(self, y_true, predictions):
        """
        Calculate the accuracy of predictions.

        :param y_true: True labels.
        :param predictions: Predictions from the model, as probabilities.
        
        :return: Accuracy as a float.
        """
        y_pred_indices = np.argmax(predictions, axis=1)
        y_true_indices = np.argmax(y_true, axis=1)
        
        accuracy = np.mean(y_pred_indices == y_true_indices)
        return accuracy
    
    def save_model(self, file_name, directory="model"):
            """
            Saves the model weights and configuration to model directory.
            """
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            model_data = {
            "final_weights": [],
            "config": {
                "layers": [{"number_of_neurons": layer.number_of_neurons,
                            "activation_function": layer.activation_function} for layer in self.layers],
                }
            }

            for weights, bias in zip(self.weights, self.bias_weights):
                bias_rounded = np.round(bias, 6)
                weights_rounded = np.round(weights,6)
                bias_reshaped = np.reshape(bias_rounded, (1, len(bias_rounded)))
                integrated_layer_weights = np.vstack([bias_reshaped, weights_rounded])
                model_data["final_weights"].append(integrated_layer_weights.tolist())

            new_file_name = "model-" + os.path.basename(file_name)
            # Save to JSON file
            with open(os.path.join(directory, new_file_name), "w") as json_file:
                json.dump(model_data, json_file)
            
            print("Model saved successfully to JSON.")

    @classmethod
    def load_model(cls, file_name, directory="model"):
        """
        Loads the model weights and configuration from model directory.
        """
        # Load configuration
        with open(os.path.join(directory, file_name), "r") as json_file:
            model_data = json.load(json_file)
        
        layers = [FFNNLayer
                  (layer_conf["number_of_neurons"], layer_conf["activation_function"])
                  for layer_conf in model_data["config"]["layers"]
                ]
        
        #  Create new instance
        classifier = cls(
            layers=layers,
            weights=[],  
        )

        classifier.weights = []
        classifier.bias_weights = []
        for integrated_weights in model_data["final_weights"]:
            np_weights = np.array(integrated_weights)
            classifier.bias_weights.append(np_weights[0, :])
            classifier.weights.append(np_weights[1:, :])  
        return classifier
    
    def printModel(self):
        for i in range(len(self.weights)):
            print("[")
            print("  ", self.bias_weights[i])
            for neuron_weight in self.weights[i]:
                print("  ", neuron_weight)
            print("], ")
    