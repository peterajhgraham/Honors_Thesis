"""
Manual Gradient Descent

A from-scratch implementation of gradient descent for a single-layer
linear network ``y = W x``. Gradients are computed analytically using
basic linear algebra rather than automatic differentiation.

Functions
---------
compute_gradient : Forward pass and analytical gradient computation.
gradient_descent : Iterative weight optimisation loop.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_gradient(W, x, y_true):
    """Compute the predicted output and analytical gradient.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape ``(1, d)``.
    x : np.ndarray
        Input vector of shape ``(d, 1)``.
    y_true : np.ndarray
        Ground-truth output of shape ``(1, 1)``.

    Returns
    -------
    y_pred : np.ndarray
        Predicted output ``W @ x``.
    gradient : np.ndarray
        Gradient ``dL/dW = (y_pred - y_true) * x^T``.
    """
    y_pred = np.dot(W, x)
    e = y_pred - y_true
    gradient = e * x.T
    return y_pred, gradient


def gradient_descent(W, x, y_true, learning_rate, epochs):
    """Perform gradient descent for a fixed number of epochs.

    Parameters
    ----------
    W : np.ndarray
        Initial weight matrix.
    x : np.ndarray
        Input vector.
    y_true : np.ndarray
        Ground-truth output.
    learning_rate : float
        Step size for weight updates.
    epochs : int
        Number of training iterations.

    Returns
    -------
    W : np.ndarray
        Optimized weight matrix.
    losses : list of float
        Loss value at each epoch.
    """
    losses = []

    for i in range(epochs):
        y_pred, gradient = compute_gradient(W, x, y_true)
        W = W - learning_rate * gradient
        loss = 0.5 * np.square(y_pred - y_true)
        losses.append(loss.item())
        print(f"Epoch {i + 1}: Loss = {loss}, Weights = {W}")

    return W, losses
