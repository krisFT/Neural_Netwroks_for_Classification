import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

"""
This script builds a two-hidden-layer neural network from scratch using Numpy for classification tasks. 
 - The hidden layers use the ReLU activation function to introduce non-linearity, 
 - and the output layer consists of a single neuron with a sigmoid activation to handle the classification. 
 - The network is trained using the squared loss function to minimize the error between predicted and true values. 
 - For simplicity, biases are excluded from the neurons.
"""

def to_one_hot(Y):
    """Convert numeric labels into binary vectors."""
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def sigmoid_derivative(x): 
    """To compute how much the output layer should adjust in Backpropagation"""
    s = sigmoid(x)
    return s * (1 - s)

def ReLU_derivative(x): 
    """To compute how much the hidden layers should adjust in Backpropagation"""
    return np.where(x > 0, 1, 0)

def forward_propagation(x, W1, W2, W3):
    """Forward pass through the network."""
    Z1 = np.dot(x, W1) 
    layer_1=ReLU(Z1)
    Z2 = np.dot(layer_1, W2) 
    layer_2 = ReLU(Z2)
    Z3 = np.dot(layer_2, W3)
    output=sigmoid(Z3)
    return output, layer_1, layer_2, Z1, Z2, Z3

def back_propagation(X, y_true, output, layer_1, layer_2, Z1, Z2, W2, W3):
    """Backward pass to compute gradients."""
    # Gradients for W3
    E3 = 2.0 * (output - y_true)  # Loss derivative
    delta3 = np.multiply(sigmoid_derivative(output), E3)  # Derivative of sigmoid
    dW3 = np.dot(layer_2.T, delta3)

    # Gradients for W2
    delta2 = np.multiply(np.dot(delta3, W3.T), ReLU_derivative(layer_2))  # Chain rule
    dW2 = np.dot(layer_1.T, delta2)

    # Gradients for W1
    delta1 = np.multiply(np.dot(delta2, W2.T), ReLU_derivative(layer_1))  # Chain rule
    dW1 = np.dot(X.T, delta1)

    return dW1, dW2, dW3

def loss(y_pred, y_true):
    """Mean squared error loss."""
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def total_loss(y_pred, y_true, weights, reg_strength=1e-5):
    """Adds L2 regularization to prevent overfitting"""
    data_loss = loss(y_pred, y_true)
    reg_loss = 0.5 * reg_strength * sum(np.sum(w * w) for w in weights)
    return data_loss + reg_loss

def predict(x, W1, W2, W3):
    _, layer_1, layer_2, _, _, _ = forward_propagation(x, W1, W2, W3)
    scores = sigmoid(np.dot(layer_2, W3))
    return scores

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

#----------------------Data Preprocessing-----------------------------#
# Import data and normalize
digits = load_digits()

# Randomly split the data
train_x, test_x, train_y, test_y = train_test_split(
    digits.data / 255., digits.target, test_size=0.1, random_state=42
)

# Convert labels to one-hot encoding if needed
train_y = np.expand_dims(train_y, axis=1)
test_y = np.expand_dims(test_y, axis=1)

train_y = to_one_hot(train_y)
test_y = to_one_hot(test_y)

#----------------------Model Configuration-----------------------------#
input_size = 64  # Number of input features
hidden_size_1 = 16  # Neurons in the first hidden layer
hidden_size_2 = 16  # Neurons in the second hidden layer
output_size = 10  # Number of classes
learning_rate = 0.001
reg = 1e-5
epochs = 3000

# Initialize weights
np.random.seed(42)
W1 = np.random.uniform(-1, 1, (input_size, hidden_size_1))  # Input to Hidden Layer 1
W2 = np.random.uniform(-1, 1, (hidden_size_1, hidden_size_2))  # Hidden Layer 1 to Hidden Layer 2
W3 = np.random.uniform(-1, 1, (hidden_size_2, output_size))  # Hidden Layer 2 to Output Layer

#----------------------Training Starts-----------------------------#
for i in range(epochs):

    # Forward propagation
    output, layer_1, layer_2, Z1, Z2, Z3 = forward_propagation(train_x, W1, W2, W3)

    # Compute loss and accuracy
    data_loss = loss(output, train_y)
    acc = accuracy(output, train_y)
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    total_loss = data_loss + reg_loss

    if i % 300 == 0:
      print(f"Iteration {i}, Loss: {total_loss:.4f}, Training Accuracy: {acc:.4f}")

    # Backpropagation
    dW1, dW2, dW3 = back_propagation(train_x, train_y, output, layer_1, layer_2, Z1, Z2, W2, W3)

    # Add regularization to gradients
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3
    
    # Update weights
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    W3 -= learning_rate * dW3

#----------------------Testing-----------------------------#
test_output = predict(test_x, W1, W2, W3)
test_accuracy = accuracy(test_output, test_y)
print(f"Testing Accuracy: {test_accuracy:.4f}")
