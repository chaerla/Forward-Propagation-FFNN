from DataPreprocessor import DataPreprocessor
from MLPClassifier import FFNNLayer, MLPClassifier

layers = [
    FFNNLayer(2, 'sigmoid'),
    FFNNLayer(4, 'sigmoid'),
]

mlp = MLPClassifier(layers, 0.5, 0.0, 1, 1)

x = [
    [-0.6, 1.6, -1.0],
    [-1.4, 0.9, 1.5],
    [0.2, -1.3, -1.0],
    [-0.9, -0.7, -1.2],
    [0.4, 0.1, 0.2]
]

y = [
    [0.41197346, 0.8314294, 0.53018536, 0.31607396],
    [0.78266141, 0.80843631, 0.55350518, 0.64278501],
    [0.58987524, 0.82160954, 0.75436518, 0.34919895],
    [0.6722004, 0.81660439, 0.59020258, 0.50870988],
    [0.47322841, 0.82808466, 0.69105452, 0.29358323]
]

TARGET_COLUMN = "Species"

preprocessor = DataPreprocessor("test_cases/iris.csv")
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
