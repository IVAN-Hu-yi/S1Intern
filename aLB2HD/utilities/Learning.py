import numpy as np


def relu(x):
    return np.maximum(0, x)


def mOSA(input, output, W_t0, lr=0.1, ff_coefficient=1, fb_coefficient=1):
    """
    Stepwise learning algorithms in Python.

    Parameters:
        name (str): Name of the learning algorithm.
        input (numpy.ndarray): Input data.
        output (numpy.ndarray): Output data.
        W_t0 (numpy.ndarray): Initial weights.
        lr (float): Learning rate.
        ff_coefficient (float): Feedforward coefficient.
        fb_coefficient (float): Feedback coefficient.

    Returns:
        numpy.ndarray: Updated weights.
    """
    diff = ff_coefficient * input - fb_coefficient * output.T @ W_t0
    W_t = W_t0 + lr * output[:, np.newaxis] @ diff
    W_t = relu(W_t)

    return W_t

