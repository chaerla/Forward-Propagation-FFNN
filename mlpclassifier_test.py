from MLPClassifier import FFNNLayer, MLPClassifier

layers = [
    FFNNLayer(2, 'sigmoid'),
]

weights = [[[0.3, 0.1], [0.2, 0.6], [0.8, 0.3]]]


expected_weights = [[[0.2329, 0.0601], [0.1288, 0.6484], [0.8376, 0.2315]]]
mlp = MLPClassifier(layers, 0.1, 0.01, 10, 2, weights, "max_iteration", expected_weights)

x = [[0.5, 0.0], [0.0, 0.5]]

y = [[0.0, 1.0], [1.0, 0.0]]

mlp.fit(x, y)