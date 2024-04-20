from MLPClassifier import FFNNLayer, MLPClassifier

layers = [
    FFNNLayer(2, 'linear'),
    FFNNLayer(2, 'relu'),
]

weights = [
            [
                [0.1, 0.2],
                [-0.3, 0.5],
                [0.4, 0.5]
            ],
            [
                [0.2, 0.1],
                [0.4, -0.5],
                [0.7, 0.8]
            ]
        ]

mlp = MLPClassifier(layers, 0.1, 0.0, 1, 2, weights)

x = [
            [-1.0, 0.2],
            [0.2, -1.0]
        ]

y = [
            [1.0, 0.1],
            [0.1, 1.0]
        ]

mlp.fit(x, y)
mlp.predict()
