import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#----------------------Data Preprocessing-----------------------------#
digits = load_digits()
X = digits.data / 255.0  # Normalize pixel values
y = to_categorical(digits.target, 10)  # One-hot encode labels
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)

#----------------------Model Configuration-----------------------------#
network = models.Sequential([
    layers.Dense(16, activation='relu', input_dim=64, #relu sigmoid
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    #layers.Dropout(0.3),  # Add dropout for regularization
    layers.Dense(16, activation='relu', #relu sigmoid
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    #layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # sigmoid softmax
])

# Compile with better optimizer and loss
#Adam
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy', #categorical_crossentropy mean_squared_error
    metrics=['accuracy']
)
# SGD
# network.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#     loss='mean_squared_error', #categorical_crossentropy mean_squared_error
#     metrics=['accuracy']
# )

# Train the model
history = network.fit(
    train_x, train_y, 
    epochs=300, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = network.evaluate(test_x, test_y, verbose=1)
print(f"Test Loss: {test_loss}", f"Test Accuracy: {test_accuracy}")

# Predict class probabilities for the test set
predictions = network.predict(test_x)
predicted_classes = predictions.argmax(axis=1)
true_classes = test_y.argmax(axis=1)

# Compare predictions with true labels for evaluation
print("First 5 Predicted Classes:", predicted_classes[:5])
print("First 5 True Classes:", true_classes[:5])

# Print summary of the network architecture
print(network.summary())

# Plot the network architecture
tf.keras.utils.plot_model(network, show_shapes=True, to_file='model_plot.png')
img = mpimg.imread('model_plot.png')
plt.imshow(img)
plt.axis('off')  # Hide axes for a cleaner look
plt.show()

# Plot the accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Testing Accuracy')  # Add test accuracy
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()