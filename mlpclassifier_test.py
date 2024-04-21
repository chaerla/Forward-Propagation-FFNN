from MLPClassifier import FFNNLayer, MLPClassifier

layers = [
    FFNNLayer(2, 'linear'),
FFNNLayer(2, 'relu'),
]

weights =  [
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


expected_weights = [
            [
                [0.08592, 0.32276],
                [-0.33872, 0.46172],
                [0.449984, 0.440072]
            ],
            [
                [0.2748, 0.188],
                [0.435904, -0.53168],
                [0.68504, 0.7824]
            ]
        ]
mlp = MLPClassifier(layers, 0.1, 0.05, 1, 2, weights, "max_iteration", expected_weights)

x = [
            [-1.0, 0.2],
            [0.2, -1.0]
        ]

y = [
            [1.0, 0.1],
            [0.1, 1.0]
        ]

mlp.fit(x, y)