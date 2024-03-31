from collections import namedtuple
from graphviz import Digraph
from utils import get_random_pale_color

import FFNN


class FFNNVisualizer:
    """
    Visualizer for Feed-Forward Neural Network (FFNN).
    """
    def __init__(self, ffnn: FFNN):
        """
        Initializes the visualizer with the given FFNN.

        :param ffnn: The FFNN to be visualized.
        """
        self.ffnn = ffnn

        # Create a mock input layer and insert it at the beginning of the layers list
        input_layer = self._create_mock_input_layer()
        self.layers = [input_layer] + self.ffnn.layers

    def _create_mock_input_layer(self):
        """
        Creates a mock input layer with the number of neurons equal to the input size of the FFNN.

        :return: A mock input layer.
        """
        MockLayer = namedtuple('InputLayer', ['number_of_neurons'])
        return MockLayer(self.ffnn.input_size)

    def visualize(self, output_path: str = 'bin/FFNN'):
        """
        Visualizes the FFNN by creating a graph with nodes and edges representing neurons and connections.

        :param output_path: The path where the output image will be saved.
        """
        dot = Digraph(format='png')
        dot.attr(ranksep='3')  # Set the distance between layers
        dot.attr(nodesep='0.5')  # Set the distance between node in one layer

        # Add nodes and edges for each layer
        for i, layer in enumerate(self.layers):
            with dot.subgraph(name=f'cluster_{i}') as c:
                c.attr(color='white')
                self._add_nodes_to_graph(c, i, layer)
                if i > 0:
                    self._add_edges_to_graph(dot, i)

        # Render the graph and save it to the output path
        dot.render(output_path, view=True)

    def _add_nodes_to_graph(self, graph, layer_index, layer):
        num_neurons = layer.number_of_neurons if layer_index == len(
            self.layers) - 1 else layer.number_of_neurons + 1

        # Get a random pale color for the layer
        color = get_random_pale_color()

        # Add a node for each neuron
        for j in range(num_neurons):
            node_name = self._get_node_name(layer_index, j)
            graph.node(node_name, fillcolor=color, style='filled')

    def _get_node_name(self, layer_index, neuron_index):
        """
        Generates a name for a node based on its layer and neuron indices.

        :param layer_index: The index of the layer.
        :param neuron_index: The index of the neuron within its layer.
        :return: The name of the node.
        """
        if layer_index == 0:
            return 'I' + str(neuron_index)
        elif layer_index == len(self.layers) - 1:
            return 'O' + str(neuron_index + 1)
        else:
            return 'H' + str(layer_index) + str(neuron_index)

    def _add_edges_to_graph(self, graph, layer_index):
        """
        Adds edges to the given graph for each connection between the neurons in the current and previous layers.

        :param graph: The graph to which edges will be added.
        :param layer_index: The index of the current layer.
        """
        is_last_layer = layer_index == len(self.layers) - 1

        # Add an edge for each connection
        for j in range(self.layers[layer_index - 1].number_of_neurons + 1):
            source_node_name = self._get_node_name(layer_index - 1, j)
            for k in range(self.layers[layer_index].number_of_neurons):
                weight = self.ffnn.weights[layer_index - 1][j][k]
                target_node_name = self._get_node_name(layer_index, k if is_last_layer else k + 1)
                graph.edge(source_node_name, target_node_name, label=str(weight))
