"""
A granular layer class connected to dRSC and HD cells

"""

# DONE: gRSC weights inihialization
# DONE: weights updating
# TODO: Go back to HDRIng to define F_HD and F_HD_p
# TODO: set HD_to_gRSC_weight
import numpy as np
from utilities.parameters import DefaultParameters
from HD.HDRing import HDattractor
from RSCModule.dRSCLayer import dRSCLayer
from utilities.FiringRate_sigmoid import firing_rate_sigmoid


class gRSCLayer:
    def __init__(self, params: DefaultParameters, HDRing: HDattractor):
        self.params = params
        # Initialize the gRSC layer
        self.initialize_layer(HDRing)
        self._initialize_weights()

    def initialize_layer(self, HDRing: HDattractor):
        self.U_gRSC = self.params.U_HD2gRSC_gain_factor * np.dot(
            HDRing.F_HD_p, np.diag(np.ones(self.params.N_bin))
        )
        self.F_gRSC = firing_rate_sigmoid(
            self.U_gRSC,
            self.params.alpha_gRSC,
            self.params.beta_gRSC,
            self.params.gamma_gRSC,
        )

    def _initialize_weights(self):
        self._initialize_gRSC_to_dRSC_weights()

    def _initialize_gRSC_to_dRSC_weights(self):
        self._gRSC_to_dRSC_weight = np.eye(self.params.N_bin) * 1

    def store_as_previous(self):
        """
        store the current state as previous state
        """
        self.U_gRSC_p = self.U_gRSC.copy()
        self.F_gRSC_p = self.F_gRSC.copy()

    def update(self, HDRing: HDattractor):
        """
        Update the gRSC layer
        """
        feedforward = (
            self.params.U_HD2gRSC_gain_factor
            * HDRing.F_HD
            @ HDRing._HD_to_gRSC_weight.T
            / self.params.N_bin
        )
        self_inhibition = (
            self.U_gRSC2gRSC
            * self.F_gRSC_p
            @ (np.ones((1, self.params.N_bin)).T / self.params.N_bin)
        )

        self.U_gRSC += self.params.decay_rate_gRSC * (
            feedforward - self_inhibition - self.U_gRSC_p
        )
        self.F_gRSC = firing_rate_sigmoid(
            self.U_gRSC,
            self.params.alpha_gRSC,
            self.params.beta_gRSC,
            self.params.gamma_gRSC,
        )

    def lr_update(self, present_time, stop_learning_time, j, dt):
        self.lr = (
            (present_time <= stop_learning_time)
            * self.params.lr_initial_rate_g2dRSC
            * np.exp(-self.params.lr_decay_rate_g2dRSC * j * dt)
        )
        self.lr_slow = (
            (present_time <= stop_learning_time)
            * self.params.lr_initial_rate_g2dRSC_slow
            * np.exp(-self.params.lr_decay_rate_g2dRSC_slow * j * dt)
        )

        self.learning_rate_g2dRSC_vector = np.concatenate(
            (
                np.full(self.params.N_bin // 2, self.lr_slow),
                np.full(self.params.N_bin // 2, self.lr),
            )
        )

    def weight_update(self, dRSC: dRSCLayer):
        # Update weights using learning rates
        self._gRSC_to_dRSC_weight = (
            self._gRSC_to_dRSC_weight
            + self.learning_rate_g2dRSC_vector
            * (dRSC.F_dRSC_p[0, :].reshape(-1, 1) @ dRSC.F_dRSC_p[0, :].reshape(1, -1))
        )
        Wg2d_total = np.sqrt(np.sum(self._gRSC_to_dRSC_weight**2, axis=1))
        Wg2d_total_metric = np.tile(Wg2d_total[:, np.newaxis], (1, self.params.N_bin))
        Wg2d_target = np.minimum(Wg2d_total, self.params.W_g2dRSC_weight_scale)
        Wg2d_target_metric = np.tile(Wg2d_target[:, np.newaxis], (1, self.params.N_bin))
        self._gRSC_to_dRSC_weight = (
            self._gRSC_to_dRSC_weight / Wg2d_total_metric * Wg2d_target_metric
        )
