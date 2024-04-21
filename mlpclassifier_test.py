from DataPreprocessor import DataPreprocessor
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
# mlp.predict(x)

TARGET_COLUMN = "Species"

preprocessor = DataPreprocessor("test_cases_mlp/iris.csv")
X_train, X_test, y_train, y_test = preprocessor.preprocess(TARGET_COLUMN)

print('XTrain')
print(X_train)

print('XTest')
print(X_test)

print('YTrain')
print(y_train)

print('YTest')
print(y_test)

mlp.fit(X_train, y_train)
# mlp.predict()

# To decode the predicted result
y_pred_encoded = [0]
y_pred = preprocessor.decode_labels(y_pred_encoded)
print(y_pred)
