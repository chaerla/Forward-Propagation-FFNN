from FFNN import FFNNLayer, FFNN
from FFNNVisualizer import FFNNVisualizer
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

      weights = data["case"]["weights"]
      input_size = data["case"]["model"]["input_size"]
      input_x = data["case"]["input"]
      expected_output = data["expect"]["output"]

      model = FFNN(input_size, layers, weights)
      model.fit(input_x, expected_output)
      model.print_expected_output()
      visualizer = FFNNVisualizer(model)
      visualizer.visualize()

      result = model.predict()
      model.print_prediction_results()

      max_sse = data["expect"]["max_sse"]
      sse = model.calculate_sse()
      print(f"Sum Squared Error: {sse:.4f}")
      if sse < max_sse:
          print("Sum Squared Error(SSE) of prediction is lower than Maximum SSE")
      else:
          print("Sum Squared Error(SSE) of prediction surpass the Maximum SSE")
    except KeyError as ke:
      print('Key', ke, "not found in json data. Please check your json data format")
    except Exception as error:
      print("An exception occurred: ", error)
