import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utilities.parameters import DefaultParameters
from utilities.vonmisespdf import vonmisespdf
from utilities.inverseFR_sigmoid import inverseFR_sigmoid
from utilities.AngularDiff import AngularDiff
from utilities.Differentiation_ascending import Differentiation_ascending


class HDattractor(DefaultParameters):

    '''
    HD attractor network
    '''
    def __init__(self, Operation_Cyaegha=0):
        '''
        n: number of neurons
        '''
        super().__init__()
        self.n = self.N_bin
        # self-connections weights of the HD neurons
        # self._get_recurrent_column_weights()
        # self._circular_matrix()
        # self.dw_t = np.zeros((self.recurrent_column_weight.shape))
        self.Operation_Cyaegha = Operation_Cyaegha

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
        self.U0 = U0
        self.recurrent_column_weight = np.zeros(self.n)

        Uf = np.fft.fft(U0)
        Ff = np.fft.fft(F0)
        Wf = (Uf * Ff) /(self._lambda + np.abs(Ff)**2)
        W_pi = np.fft.ifft(Wf) # inverse fourier transform 

        for k in range(self.N_bin):
            idx = np.where(self.Angle == AngularDiff(self.Angle[k], 180))
            self.recurrent_column_weight[idx] = W_pi[k]
        
        return self.recurrent_column_weight

    def _dw_HD(self):
        self.dw_t = Differentiation_ascending(self.recurrent_column_weight, self.angle_gap) + self.weight_noise_scale * np.random.randn(self.N_bin)

    def get_aneular_velocity(self, trajectory):
        return Differentiation_ascending(trajectory, self.dt) 

    def get_angular_acceleration(self, trajectory):
        return Differentiation_ascending(self.get_aneular_velocity(trajectory), self.dt)
        
    def get_position(self):
        pass

    def _compute_gamma(self, velocity, acceleration):

        '''
        compute the gamma term in the HD attractor network '''
        gamma = - self.time_constant_HD * (velocity + self.rho * acceleration)
        gamma = gamma + self.asyweightstrength_noise_scale * np.random.randn(self.T_len) # add noise

        return gamma
    
    def _recurrent_weight_update(self, gamma, trajectory):
        '''
        update the recurrent weights of the HD neurons
        '''
        
        self.dw = self._dw_HD()
        self.recurrent_column_weight -= self.dt * (self.dw * self.time_constant_HD * self.get_aneular_velocity(trajectory) )


    def _get_dRSC2HD_info(self, dRSCLayer):
        '''Return dRSc2HD weight and activity

        U_dRSC = g_dRSC2HD/n_HD * W_dRSC2HD * f_dRSC

        Returns:
        W_dRSC2HD: dRSC to HD weight
        U_dRSC: dRSC activity
        ''' 
        pass
    
    def trajectory_settings(self):

        self._hd_trajectory_Noatun()
        self._hd_trajectory_Niflheim()

    def set_HD_trajectory(self, trajectory):
        '''
        set the trajectory of the HD neurons
        '''

        self.real_hdTrajectory = trajectory['HD']
        self.trajectory = np.zeros(trajectory['HD'].shape)
        self.trajectory_settings()

    def _hd_trajectory_Niflheim(self):
        '''
        return the trajectory of the HD neurons
        '''
        initiation = np.round(self.beginning / self.dt_exp)
        self.trajectory = self.real_hdTrajectory[int(initiation): int(self.T_len + initiation)+1]
        trajectory_0 = self.real_hdTrajectory[0]
        self.real_hdTrajectory[0] = 0

        for i in range(1, self.T_len):
            self.real_hdTrajectory[i] = AngularDiff(self.real_hdTrajectory[i], trajectory_0)
        
            # check if the angle between consecutive points is larger than 30

            if np.abs(AngularDiff(self.real_hdTrajectory[i], self.real_hdTrajectory[i-1])) > 30:
                trajectory_0 += AngularDiff(self.real_hdTrajectory[i], self.real_hdTrajectory[i-1])
                self.real_hdTrajectory[i] = self.real_hdTrajectory[i-1]
    
    # def _hd_trajectory_Niflheim_no(self):
    #     self.real_hdTrajectory[0] = 0
    #     for i in range(0, self.T_len-1):
    #         current_moment = i*self.dt_exp

    #         if current_moment > self.stop_learning_time:
    #             vlcty = 360/(self.time - self.stop_learning_time)
    #         elif np.mod(np.round(current_moment, 2)):
    #             vlcty = self.vlcty_tuning
    #         else:
    #             vlcty = 0
    #         self.real_hdTrajectory[i+1] = AngularDiff(self.real_hdTrajectory[i] + vlcty * self.dt_exp, 0)
    
    def _hd_trajectory_Noatun(self):
        ''' poisoning scenario '''
        ppt_hd = (np.random.rand(1, self.N_env) * 360 - 180) * self.Operation_Cyaegha # random head movement data for a poisoning scenario

        poisoned_hd = 0
        sample_gap = np.int64(np.round(self.dt_exp/self.dt))

        for i in range(0, self.T_len-1):
            pre_HD = AngularDiff(self.real_hdTrajectory[i]+poisoned_hd, 0)
            post_HD = AngularDiff(self.real_hdTrajectory[i+1]+poisoned_hd, 0)

            # adjust for angles larger than 180

            while abs(pre_HD - post_HD) > 180:
                if pre_HD < post_HD:
                    post_HD -= 360
                else:
                    post_HD += 360

            # calculate detailed trajectory based on adjusted angles
            for j in range(sample_gap):
                current_idx = (i-1)*sample_gap + j
                self.trajectory[current_idx] = AngularDiff(pre_HD + (post_HD - pre_HD) * 1.0/(sample_gap*(j-1)), 0)
            
            if np.where(self.time_CueShifting >= (self.beginning + (i+1)*self.dt_exp))[0] != np.where(self.time_CueShifting >= (self.beginning + i*self.dt_exp))[0]:
                poisoned_hd = ppt_hd[np.where(self.time_CueShifting >= (self.beginning + i*self.dt_exp))[0]]

        self.trajectory[current_idx+1] = AngularDiff(post_HD, 0)
        self.time = np.arange(0, self.time, self.dt)
        self.T_len = len(self.time)

    # def _hd_trajectory_Noatun_no(self):
    #     ''' no poisoning scenario '''
    #     self.trajectory = self.real_hdTrajectory.copy()




