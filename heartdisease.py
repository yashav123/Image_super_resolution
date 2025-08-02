import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the data
file_path = 'framingham.csv'
data = pd.read_csv(file_path)

# Drop rows with missing values
data_clean = data.dropna()

# Define features and target
features = data_clean.drop(columns=['TenYearCHD'])
target = data_clean['TenYearCHD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Add intercept term to the feature matrix
def add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

X_train = add_intercept(X_train.values)
X_test = add_intercept(X_test.values)
y_train = y_train.values
y_test = y_test.values

# Sigmoid function with clipping to prevent overflow
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1/m) * X.T @ (sigmoid(X @ theta) - y)
        theta -= learning_rate * gradient
    return theta

# Initialize parameters
theta = np.zeros(X_train.shape[1])
learning_rate = 0.01
iterations = 10000

# Train the model using gradient descent
theta = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Make predictions
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

y_pred_train = predict(X_train, theta)
y_pred_test = predict(X_test, theta)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

TN, FP, FN, TP = conf_matrix.ravel()

# Calculate metrics
accuracy = test_accuracy * 100
false_negatives_percentage = (FN / (FN + TP)) * 100 if (FN + TP) > 0 else 0
false_positives_percentage = (FP / (FP + TN)) * 100 if (FP + TN) > 0 else 0

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {accuracy:.2f}%")
print(f"False Negatives: {false_negatives_percentage:.2f}%")
print(f"False Positives: {false_positives_percentage:.2f}%")
