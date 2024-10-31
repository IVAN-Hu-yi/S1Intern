
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utilities.parameters import DefaultParameters
from utilities.vonmisespdf import vonmisespdf
from utilities.inverseFR_sigmoid import inverseFR_sigmoid
from utilities.AngularDiff import AngularDiff



class HDattractor(DefaultParameters):

    '''
    HD attractor network
    '''


    def __init__(self, n, operation_runenschrift:bool = False):
        '''
        n: number of neurons
        '''
        self.n = n 
        self.operation_runenschrift = operation_runenschrift

        # self-connections weights of the HD neurons
        self._circular_matrix()


        super().__init__()

    

    def _circular_matrix(self):
        '''
        return a rotation matrix R
        '''
        R = np.zeros((self.n, self.n))
        R[0, -1] = 1
        R[1:, :self.n-1] = np.eye(self.n-1)
        self.rotations = [np.identity(self.n)]
        for i in range(1, self.n):
            self.rotations.append(np.linalg.matrix_power(R, i))
    
    def _circular_shift(self, x):
        '''
        circular shift of the vector x and return a weight matrix W

        x: estimated column vector of W_HD
        '''
        x = x.reshape(-1, 1)
        columns = [x]
        for i in range(1, self.n):
            columns.append(self.rotations[i] @ x)
        W = np.hstack(columns)

        return W
        

    def _get_recurrent_column_weights(self):
        '''
        pre-wired self-connections weights of the HD neurons

        This function used a scaled von Mises distribution as circular Gaussian distribution for the idealized firing rate of the HD neurons, prarameterized by the weight_bias and precision_HD, and maximum firing as well.

        This function used such firing rates, with the help of flourier transform, to approximate the solution for the recurrent weights of the HD neurons.
        '''
        # Firing rates of the HD neurons


        F0 = vonmisespdf(self.Angle, self.weight_bias, self.precision_HD)
        F0 = (F0 / F0.max()) * 0.8

        # get activation level from the firing rate
        U0 = inverseFR_sigmoid(F0, self.alpha_HD, self.beta_HD, self.gamma_HD)
        self.recurrent_column_weight = np.zeros(self.n)


        if self.operation_runenschrift: # ??? seem very different from the journal paper
            Uf = np.fft.fft(U0)
            Ff = np.fft.fft(F0)
            Wf = (Uf * Ff) /(self._lambda + np.abs(Ff)**2)
            W_pi = np.fft.ifft(Wf) # inverse fourier transform 

            for k in range(self.N_bin):
                idx = np.where(self.Angle == AngularDiff(self.Angle[k], 180))
                self.recurrent_column_weight[idx] = W_pi[k]
        else: # ??? 
            self.recurrent_column_weight = 100 * (vonmisespdf(self.Angle, self.weight_bias, 5) - 0.5 * vonmisespdf(self.Angle, self.weight_bias, 0.2) - 0.002)
        
        return self.recurrent_column_weight

    def get_angular_velocity(self):
        pass

    def get_angular_acceleration(self):
        pass

    def get_position(self):
        pass

    def _compute_gamma(self, velocity, acceleration):

        '''
        compute the gamma term in the HD attractor network
        '''
        
        gamma = - self.time_constant_HD * (velocity + self.rho * acceleration)
        gamma = gamma + self.asyweightstrength_noise_scale * np.random.randn(self.T_len) # add noise

        return gamma
    
    def _recurrent_weight_update(self, gamma):
        '''
        update the recurrent weights of the HD neurons
        '''
        
        return self.recurrent_column_weight - self.time_constant_HD * self.get_angular_velocity()*self.get_angular_acceleration()