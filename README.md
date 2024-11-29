# Neural Networks for Classification

## Project Overview
This project compares the performance of two neural networks for a classification task on MNIST dataset:
1. A **neural network built from scratch** using only NumPy (`NN_for_Classification.py`).
2. A **neural network implemented with TensorFlow** (`Classification_with_TensorFlow.py`).

The goal is to analyze and compare the performance of these implementations under different configurations, including activation functions, loss functions, and optimizers. Results from training and testing are reported for each configuration.

---

## Methodology

### 1. **Custom Implementation (`NN_for_Classification.py`)**
- **Architecture**:
  - Two hidden layers using **ReLU** as the activation function.
  - Output layer uses **Sigmoid** for binary classification.
- **Training Process**:
  - Forward propagation.
  - Backpropagation for gradient computation.
  - Weights updated using gradient descent.
  - Loss computed using **Mean Squared Error (MSE)** for simplicity.
- **Epoch**: The model was trained for 3000 epochs.
- **Performance**:
  - Training Accuracy: **99.94%**
  - Testing Accuracy: **97.78%**

### 2. **TensorFlow Implementation (`Classification_with_TensorFlow.py`)**
- The network mimics the custom implementation initially but is extended to conduct an **ablation study**. Variations in activation functions, optimizers, and loss functions are explored:
  1. **Activation Functions**:
     - Replace **ReLU** with **Sigmoid** for hidden layers.
     - Replace **Sigmoid** with **Softmax** for the output layer.
  2. **Optimizers**:
     - Compare **SGD** and **Adam** optimizers.
  3. **Loss Functions**:
     - Compare **Mean Squared Error (MSE)** and **Cross-Entropy**.
- **Epoch**: The model was trained for 300 epochs.
- **Performance**: Detailed results are available in the Experiment Results section.

---

## Experiment Results

### **Baseline (ReLU + Sigmoid)**
| Optimizer | Loss Function | Training Accuracy | Testing Accuracy |
|-----------|---------------|-------------------|------------------|
| Adam      | Cross-Entropy | 98.90%           | 94.99%          |
| Adam      | MSE           | 80.69%           | 78.33%          |
| SGD       | Cross-Entropy | 17.53%           | 12.77%          |
| SGD       | MSE           | 10.45%           | 6.11%           |

---

### **Sigmoid for Hidden Layers (Sigmoid + Sigmoid)**
| Optimizer | Loss Function | Training Accuracy | Testing Accuracy |
|-----------|---------------|-------------------|------------------|
| Adam      | Cross-Entropy | 88.04%           | 86.67%          |
| Adam      | MSE           | 9.69%            | 10.56%          |
| SGD       | Cross-Entropy | 10.52%           | 10.55%          |
| SGD       | MSE           | 10.03%           | 4.44%           |

---

### **Softmax for Output Layer (ReLU + Softmax)**
| Optimizer | Loss Function | Training Accuracy | Testing Accuracy |
|-----------|---------------|-------------------|------------------|
| Adam      | Cross-Entropy | 99.18%           | 96.67%          |
| Adam      | MSE           | 92.16%           | 90.56%          |
| SGD       | Cross-Entropy | 20.82%           | 20.56%          |
| SGD       | MSE           | 12.78%           | 10.56%          |

---

### **Softmax for Output Layer (Sigmoid + Softmax)**
| Optimizer | Loss Function | Training Accuracy | Testing Accuracy |
|-----------|---------------|-------------------|------------------|
| Adam      | Cross-Entropy | 88.87%           | 89.44%          |
| Adam      | MSE           | 10.10%           | 6.11%           |
| SGD       | Cross-Entropy | 10.58%           | 10.56%          |
| SGD       | MSE           | 10.52%           | 10.56%          |

---

## Conclusion
- **Custom Implementation**: Achieved high accuracy with a straightforward two-hidden-layer neural network using ReLU and Sigmoid activations and MSE as the loss function.
- **TensorFlow Implementation**:
  - Best performance observed with **ReLU + Softmax** using **Adam optimizer** and **Cross-Entropy loss**.
  - Replacing ReLU with Sigmoid for hidden layers significantly reduced accuracy.
  - Changing the loss function to MSE generally degraded performance, especially with SGD.

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/krisFT/Neural_Netwroks_for_Classification.git
2. Run the custom implementation:
   ```bash
   python NN_for_Classification.py
3. Run the TensorFlow implementation:
   ```bash
   python Classification_with_TensorFlow.py
