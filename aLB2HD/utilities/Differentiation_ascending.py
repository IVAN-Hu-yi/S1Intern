import numpy as np
from utilities.AngularDiff import AngularDiff

def Differentiation_ascending(x, dt):
    '''
    ascending differentiation of the vector x

    Parameters
    ----------
    x : np.array
        _description_
    dt : _type_
        _description_
    '''
    dx = np.zeros(x.shape)
    for i in range(1, x.shape[0]):
        dx[i] = (AngularDiff(x[i-1], x[i]) * 1) / dt
    dx[-1] = dx[-2]

    return dx
