from sklearn.neural_network import MLPClassifier
from DataPreprocessor import DataPreprocessor

# Load the Iris dataset 
preprocessor = DataPreprocessor("test_cases_mlp/iris.csv")

# Split the data into training and test sets 
X_train, X_test, y_train, y_test = preprocessor.preprocess("Species")
  
# Define the model with mini-batch gradient descent 
model = MLPClassifier(hidden_layer_sizes=(2,2),learning_rate='constant', learning_rate_init=0.1, alpha=0.00001, solver='sgd', batch_size=50, max_iter=100) 
  
# Train model
model.fit(X_train, y_train) 

print(model.predict(X_test))
  
# Evaluate model
score = model.score(X_test, y_test) 
print(score) 