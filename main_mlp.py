from MLPClassifier import FFNNLayer, MLPClassifier
import json

if __name__ == '__main__':
    file_path = input("Enter json file path: ")
    f = open(file_path)
    data = json.load(f)

    try:
      data_layers = data["case"]["model"]["layers"]
      layers = []
      for layer in data_layers:
        activation_func = layer["activation_function"]
        if activation_func not in ["linear", "relu", "sigmoid", "softmax"]:
          raise Exception("Activation function " + activation_func + " not available")
        layers.append(FFNNLayer(layer["number_of_neurons"], activation_func))

      weights = data["case"]["initial_weights"]
      input_size = data["case"]["model"]["input_size"]
      X_train = data["case"]["input"]
      y_train = data["case"]["target"]
      learning_rate = data["case"]["learning_parameters"]["learning_rate"]
      batch_size = data["case"]["learning_parameters"]["batch_size"]
      max_iteration = data["case"]["learning_parameters"]["max_iteration"]
      error_threshold = data["case"]["learning_parameters"]["error_threshold"]

      model = MLPClassifier(layers, learning_rate, error_threshold, max_iteration, batch_size, weights)

      model.fit(X_train, y_train)

      expected_weights = data["expect"]["final_weights"]
      expected_stopped_by = data["expect"]["stopped_by"]
      # Print weight and stopped by, bandingin sama expected

      sse = model.calculate_sse(expected_weights)
      # print(f"Sum Squared Error: {sse:.4f}")
      # if sse < 1e-7:
      #     print("Sum Squared Error(SSE) of prediction is lower than Maximum SSE")
      # else:
      #     print("Sum Squared Error(SSE) of prediction surpass the Maximum SSE")
    except KeyError as ke:
      print('Key', ke, "not found in json data. Please check your json data format")
    except Exception as error:
      print("An exception occurred: ", error)
