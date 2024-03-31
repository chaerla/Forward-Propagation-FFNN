from FFNN import FFNNLayer, FFNN
from FFNNVisualizer import FFNNVisualizer

if __name__ == '__main__':
    # to do ask for json filename
    layers = [
        FFNNLayer(4, 'relu'),
        FFNNLayer(3, 'relu'),
        FFNNLayer(2, 'relu'),
        FFNNLayer(1, 'sigmoid')
    ]
    weights = [
        [
            [0.1, 0.2, 0.3, -1.2],
            [-0.5, 0.6, 0.7, 0.5],
            [0.9, 1.0, -1.1, -1.0],
            [1.3, 1.4, 1.5, 0.1]
        ],
        [
            [0.1, 0.1, 0.3],
            [-0.4, 0.5, 0.6],
            [0.7, 0.4, -0.9],
            [0.2, 0.3, 0.4],
            [-0.1, 0.2, 0.1]
        ],
        [
            [0.1, 0.2],
            [-0.3, 0.4],
            [0.6, 0.1],
            [0.1, -0.4]
        ],
        [[0.1], [-0.2], [0.3]]
    ]
    model = FFNN(3, layers, weights)
    model.fit([[-1.0, 0.5, 0.8]])

    visualizer = FFNNVisualizer(model)
    visualizer.visualize()

    # to do: rapiin output
    print(model.predict())