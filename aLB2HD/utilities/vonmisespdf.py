import numpy as np
from scipy.special import i0 # Importing the modified Bessel function of the first kind, order 0

def vonmisespdf(x, mu, k):
    """ 
    von Mises distribution with mean (mu) and spread kappa (k) parameters. 
    Computations done in log space to allow much larger k's without overflowing.
    """
    
    # Convert degrees to radians by multiplying by pi/180
    x_rad = np.deg2rad(x)
    mu_rad = np.deg2rad(mu)
    
    # Compute the PDF in log space
    log_term = k * np.cos((np.pi / 180) * (x_rad - mu_rad))
    constant_term = np.log(360) + np.log(i0(k)) + k
    
    # Compute the exponential of the log term minus the constant term
    y = np.exp(log_term - constant_term)
    
    return y