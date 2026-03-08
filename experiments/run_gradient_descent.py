"""
Experiment: Manual Gradient Descent on a Single-Layer Linear Network

Trains a weight matrix W to minimise the squared error between W*x and
y_true over 100 epochs, plotting the loss curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.gradient_descent.gradient_descent import gradient_descent
from src.visualization.surface_3d import visualize_3d_surface

# --- Define problem ---
x = np.array([[1], [2], [3]])       # Input vector  (3x1)
W = np.array([[0.2, 0.4, 0.6]])     # Initial weights (1x3)
y_true = np.array([[1]])             # Target output
learning_rate = 0.01
epochs = 100

# --- Train ---
optimized_W, losses = gradient_descent(W, x, y_true, learning_rate, epochs)

# --- Plot loss curve ---
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Training Iterations/Epochs")
plt.grid(True)
plt.show()

print("Final optimized weights:", optimized_W)

# --- 3-D surface visualisation ---
visualize_3d_surface()
