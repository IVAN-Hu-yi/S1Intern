"""
A dysgranular layer class connected to gRSC and HD cells

its connection srength with HD is not updated during the simulation
"""

import numpy as np
from utilities.parameters import DefaultParameters
from aLBmodule.aLBCells import ALBcells
from utilities.FiringRate_sigmoid import firing_rate_sigmoid
from RSCModule.gRSCLayer import gRSCLayer


class dRSCLayer:
    def __init__(self, params: DefaultParameters):
        self.params = params

        self.initialize_layer()

    def initialize_layer(self):
        self.U_dRSC = np.zeros(self.params.N_bin)
        self.F_dRSC = np.zeros(self.params.N_bin)

    def store_as_previous(self):
        """
        store the current state as previous state
        """
        self.U_dRSC_p = self.U_dRSC.copy()
        self.F_dRSC_p = self.F_dRSC.copy()

    def weights_initialization(self):
        self._dRSC_to_dRSC_weight = np.ones(self.params.N_bin) / self.params.N_bin
        self._dRSC_to_HD_weight = (
            np.eye(self.params.N_bin)
            - np.ones((self.params.N_bin, self.params.N_bin))
            / self.params.U_dRSC2HD_i_gain_factor
        )

    def update(self, aLBLayer: ALBcells, gRSC: gRSCLayer):
        """
        Update the dRSC layer
        """

        selfInhibition = (
            self.params.U_dRSC2dRSC_gain_factor
            * self.F_dRSC_p
            @ self._dRSC_to_dRSC_weight.T
        )

        aLBComponent = (
            self.params.U_arep2dRSC_gain_factor
            * aLBLayer.F_arep_p
            @ aLBLayer._aLB_to_dRSC_weight.T
        )

        gRSCComponent = (
            self.params.U_arep2dRSC_gain_factor
            * gRSC.F_gRSC_p
            @ gRSC._gRSC_to_dRSC_weight.T
        )

        self.U_dRSC = self.U_dRSC_p + self.params.decay_rate_dRSC * (
            aLBComponent + gRSCComponent - selfInhibition - self.U_dRSC_p
        )

        self.F_dRSc = firing_rate_sigmoid(
            self.U_dRSC_p,
            self.params.alpha_dRSC,
            self.params.beta_dRSC,
            self.params.gamma_dRSC,
        )
