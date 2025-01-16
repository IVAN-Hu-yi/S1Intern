'''
    A granular layer class connected to dRSC and HD cells
'''

import numpy as np
from utilities.parameters import DefaultParameters
from utilities.FiringRate_sigmoid import firing_rate_sigmoid


class gRSCLayer():

    def __init__(self, params, HDRing):

        self.params = params
        self.HDRing = HDRing
       # Initialize the gRSC layer
        self.initialize_layer() 

        self._gRSC_to_dRSC_weight = np.eye(self.params.N_bin) * 1

    def initialize_layer(self):

        self.U_gRSC = self.params.U_HD2gRSC_gain_factor * self.HDRing.F_HD @ np.eye((1, self.params.N_bin)) 
        self.F_gRSC = firing_rate_sigmoid(self.U_gRSC, self.params.alpha_gRSC, self.params.beta_gRSC, self.params.gamma_gRSC)
    def store_as_previous(self):
        '''
        store the current state as previous state
        '''
        self.U_gRSC_p = self.U_gRSC.copy()
        self.F_gRSC_p = self.F_gRSC.copy()

    def update(self):
        '''
        Update the gRSC layer
        '''
        feedforward = self.params.U_HD2gRSC_gain_factor * self.HDRing.F_HD @ self.HDRing._HD_to_gRSC_weight.T/self.params.N_bin
        self_inhibition = self.U_gRSC2gRSC * self.F_gRSC_p @ ( np.ones((1, self.params.N_bin)).T/self.params.N_bin )

        self.U_gRSC += self.params.decay_rate_gRSC * (feedforward - self_inhibition - self.U_gRSC_p)
        self.F_gRSC = firing_rate_sigmoid(self.U_gRSC, self.params.alpha_gRSC, self.params.beta_gRSC, self.params.gamma_gRSC)

    

    