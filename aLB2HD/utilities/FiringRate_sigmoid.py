import numpy as np

def firing_rate_sigmoid(activation, alpha, beta, gamma):
    """
    Sigmoid function for neural activation in Python.
    
    Parameters:
        activation (array-like or scalar): Input activation values.
        alpha (float): Alpha parameter for shifting the sigmoid curve.
        beta (float): Beta parameter controlling the slope of the sigmoid.
        gamma (float): Gamma parameter for thresholding the firing rate.
    
    Returns:
        numpy.ndarray: Firing rate after applying the sigmoid function and thresholding.
    """
    # Compute the sigmoid-like activation using tanh
    firing_rate = np.tanh(beta * (activation - alpha))
    
    # Apply thresholding based on gamma
    threshold = np.tanh(beta * (gamma - alpha))
    firing_rate[firing_rate <= threshold] = 0
    
    return firing_rate