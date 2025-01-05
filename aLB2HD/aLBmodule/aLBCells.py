import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utilities.parameters import DefaultParameters
from visionModule.visualCells import visualCells
from utilities.FiringRate_sigmoid import firing_rate_sigmoid
from utilities.Learning import mOSA
from HD.HDRing import HDattractor
from tqdm import tqdm


class ALBcells():

    def __init__(self, nb, params:DefaultParameters):
        self.params = params

        self.nb = nb

        self._generate_theta_ff()
        self._generate_theta_fb()
        self.initialize_arrays()
        self._initialised_vis2aLB_weight()
        self._initialised_lateral_inhibition_weight()
        self.store_as_previous()

        # lr
        self.lr = self.params.lr_initial_rate_visual

    def _generate_theta_ff(self):
        '''
        generate feedforward oscillation of theta wave
        '''
        time_steps = np.arange(1, self.params.T_len + 1)  # Time steps
        self.theta_ff = (1 - self.params.theta_intensity_ff / 2 + 
                    self.params.theta_intensity_ff / 2 * 
                    np.sin(np.deg2rad(self.params.theta_phase_ff) + 
                           2 * np.pi * self.params.theta_frequency_ff * self.params.dt * time_steps))
    
    def _generate_theta_fb(self):
        time_steps = np.arange(1, self.params.T_len + 1)  # Create time steps (1 to T_len)
        self.theta_fb = (1 - self.params.theta_intensity_fb / 2 + 
                    self.params.theta_intensity_fb / 2 * 
                    np.sin(np.deg2rad(self.params.theta_phase_fb) + 
                           2 * np.pi * self.params.theta_frequency_fb * self.params.dt * time_steps))
    
    def initialize_arrays(self):
        # Initialize arrays using the class parameters and store as attributes
        self.U_arep = np.zeros(self.params.N_abstract)      # Abstract representation units
        self.F_arep = np.zeros(self.params.N_abstract)      # Feature abstract units
        
    def _initialised_vis2aLB_weight(self):

        # weight initialization from visual to abstract layer
        # uniorm distribution

        self.W_vis2aLB = np.random.rand(self.params.N_abstract, self.params.N_input)
        tot = np.sum(self.W_vis2aLB**2, axis=1) 
        self.W_vis2aLB = self.W_vis2aLB / tot[:, None] * self.params.Wv_weight_scale

    def _initialised_lateral_inhibition_weight(self):
        self.Inhibition_U_arep = self.params.Inhibition_U_arep

    def store_as_previous(self):
        self.U_arep_p = self.U_arep.copy()
        self.F_arep_p = self.F_arep.copy()

    def update_U_arep(self, VisualModule:visualCells, curr_time:int):
        '''
        Update the abstract representation units
        '''

        # Update the abstract representation units
        visualComponent = self.params.Uv_gain_factor * VisualModule.neural_attention * VisualModule.F_visual @ self.W_vis2aLB.T

        modulatedVisualComponent = self.theta_ff[curr_time] * visualComponent
        lateralInhibition = self.params.inhibition_U_arep * self.F_arep @ self.Inhibition_U_arep.T
        
        modulatedVisualComponent = modulatedVisualComponent.flatten()

        self.U_arep += self.params.decay_rate_arep * (self.theta_ff[curr_time] * modulatedVisualComponent - lateralInhibition - self.U_arep)

        self.curr_time = curr_time

    def update_F_arep(self):
        '''
        Update the feature abstract units
        '''
        self.F_arep = firing_rate_sigmoid(self.U_arep, self.params.alpha_arep, self.params.beta_arep, self.params.gamma_arep)

    def vis2aLB_weight_update(self, VisualModule:visualCells, curr_time:int):
        '''
        Update the weight from visual to abstract layer
        '''
        self.W_vis2aLB *= (1 - self.params.dr_weight_visual)
        self.W_vis2aLB = mOSA(self.W_vis2aLB, self.lr, self.F_arep, VisualModule.F_visal_p, self.theta_ff[curr_time], self.theta_fb[curr_time])
    
    def update_lr(self, curr_time, present_time_global, stop_learning_time):
        '''
        update the learning rate
        '''
        if present_time_global <= stop_learning_time:
            self.lr = self.params.lr_initial_rate_visual * np.exp(-self.params.lr_decay_rate_visual * curr_time * self.params.dt)




        
        