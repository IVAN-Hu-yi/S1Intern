import numpy as np

def AngularDiff(x, y):
    '''
    Compute the angular difference between two angles x and y
    '''
    
    criterion = 180
    diff = x - y

    while (diff < (0 - criterion)) or (diff >= criterion):
        if diff < (0 - criterion):
            diff = diff + 2 * criterion
        else:
            diff = diff - 2 * criterion
    
    return diff
