import numpy as np

def inverseFR_sigmoid(FRate, alpha, beta, gamma):
    '''
    Inverse function of the sigmoid function
    '''
    activation = np.log((1/FRate) -1)/(-2 * beta) + alpha
    activation[activation < gamma] = 0
    return activation
