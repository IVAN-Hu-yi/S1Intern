'''
    A granular layer class connected to dRSC and HD cells
'''

import numpy as np
from utilities.parameters import DefaultParameters



class gRSCLayer():

    def __init__(self, params):

        self.params = params
        
        self.U_gRSC = np.zeros((self.params.N_bin))
        self.F_gRSC = np.zeros((self.params.N_bin))

        self._gRSC_to_dRSC_weight = np.eye(self.params.N_bin) * 1
