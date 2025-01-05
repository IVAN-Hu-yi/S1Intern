import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utilities.parameters import DefaultParameters
from utilities.vonmisespdf import vonmisespdf
from utilities.AngularDiff import AngularDiff

vAngularDiff = np.vectorize(AngularDiff)

class visualInputs():

    def __init__(self, N_bin, N_cue, Angle, precision_visual1, precision_visual2, proximal_length, Fv_max_factor):

        self.F_visual_temp = np.zeros((1, N_bin))
        self.F_visual_feature = np.zeros((N_cue, N_bin))
        self.F_visual_temp = vonmisespdf(Angle, 0, precision_visual1)
        self.F_visual_temp = self.F_visual_temp/np.max(self.F_visual_temp) * Fv_max_factor
        self.F_visual_mean = np.mean(self.F_visual_temp)

        # edit feature cue
        self._add_blue_cue(Angle, precision_visual2, proximal_length)
        self._add_red_cue(Angle, precision_visual2)
        if N_cue > 2:
            self._add_salient_green_cue(Angle, precision_visual1)

        # nomalize the feature cue 
        F_visual_feature_max = np.max(self.F_visual_feature, axis=1, keepdims=True)
        self.F_visual_feature = (Fv_max_factor / F_visual_feature_max) * self.F_visual_feature
        self.F_visual_feature_mean = np.mean(self.F_visual_feature, axis=1, keepdims=True)
        self.F_visual_feature = (self.F_visual_mean / self.F_visual_feature_mean) * self.F_visual_feature

        # Compute the autocorrelation
        F_visual_feature_autocorr = np.sum(self.F_visual_feature.T**2, axis=0) / self.F_visual_feature.shape[1]
        # Compute the norm
        self.F_visual_feature_norm = np.sqrt(F_visual_feature_autocorr)
        

    def _add_salient_green_cue(self, angle, precision_visual1):
        self.F_visual_feature[2, :] = vonmisespdf(angle, 0, precision_visual1)

    def _add_red_cue(self, angle, precision_visual2):
        self.F_visual_feature[0, :] = vonmisespdf(angle, 90, precision_visual2) + vonmisespdf(angle, -90, precision_visual2)
    
    def _add_blue_cue(self, angle, precision_visual2, proximal_length):
        p_values = np.arange(-proximal_length / 2, proximal_length / 2 + 1)
        for p in p_values:
            self.F_visual_feature[1, :] += vonmisespdf(angle, p, precision_visual2)
        # self.F_visual_feature[1, :] /= len(p_values)
    


class visualCells(DefaultParameters):
    def __init__(self, params:DefaultParameters):
        self.params = params
        
        self.F_visual = np.zeros(self.params.N_input)
        self._initialize_visual_field()
        self._initialize_feature_attention()
        self._initialize_neural_attention()
        self.lr = self.params.lr_initial_rate_visual

        self.Inputs = visualInputs(
            self.params.N_bin,
            self.params.N_cue,
            self.params.Angle,
            self.params.precision_visual1,
            self.params.precision_visual2,
            self.params.proximal_length,
            self.params.Fv_max_factor)
        
        self.noise = self._add_visual_noise()


    def _initialize_visual_field(self):
        '''
        initialize the visual firing rate 
        '''
        self.visual_field = vonmisespdf(self.params.Angle, 0, self.params.precision_visualfield)
        self.visual_field = self.visual_field / np.max(self.visual_field)
        self.cue = np.zeros(self.params.N_cue)              # Cue array
        self.strength = np.zeros((self.params.N_cue))         # Cue strengths
        self.F_visual = np.zeros((1, self.params.N_input))

    def _initialize_feature_attention(self):
        '''
        initialize the feature attention
        '''
        self.feature_attention = np.ones(self.params.N_cue)/ (self.params.N_cue * self.params.total_featural_attention)

    def _initialize_neural_attention(self):
        '''
        initialize the neural attention
        '''
        self.neural_attention = np.zeros(self.params.N_cue * self.params.N_bin)
        for k in range(self.params.N_cue):
            self.neural_attention[k * self.params.N_bin: (k + 1) * self.params.N_bin] = self.feature_attention[k]

    def store_as_previous(self):
        '''
        store the current visual field as the previous visual field
        '''
        self.F_visal_p = self.F_visual.copy()

    def update_F_visual(self, trajectory, present_env, curr_time): 
        '''
        update the visual field
        '''
        self.cue = vAngularDiff(self.params.Cue_global[present_env, :], trajectory[curr_time]) 
        self.strength = self.params.Strength_global[present_env, :]
        
        shift = np.round(self.cue/self.params.angle_gap, 0).astype(int)
        F_visual_shift = np.roll(self.Inputs.F_visual_feature, shift, 1) + self.noise
        F_visual_shift[F_visual_shift<0] = 0

        # shape handling
        strength_expand = np.tile(self.strength, (self.params.N_bin, 1)).T
        strength_expand = strength_expand.flatten()
        strength_expand = strength_expand.reshape(1, strength_expand.shape[0])
        F_visual_shiftc = F_visual_shift.flatten()
        F_visual_shiftc = F_visual_shiftc.reshape(1, F_visual_shiftc.shape[0])

        self.F_visual = F_visual_shiftc * strength_expand

    def _add_visual_noise(self):

        '''
        add visual noise to the visual field
        '''
        noise = 2*np.random.rand(1, self.params.N_bin) - 1
        noise = noise/np.sum(np.abs(noise)) * self.params.visual_noise_intensity * self.params.N_bin

        return noise

def main():
    params = DefaultParameters()
    params.N_cue = 3
    params.N_bin = 360
    params.N_env = 10
    params.N_input = params.N_cue * params.N_bin
    n = visualCells(params)
    visualFeatures = n.Inputs.F_visual_feature
    colors = ['r', 'b', 'g']

    import matplotlib.pyplot as plt
    print(n.Inputs.F_visual_feature_norm)
    print('initialisation success')
    
    for i in range(visualFeatures.shape[0]):
        plt.plot(visualFeatures[i, :], label=f'{i}', color=colors[i])
    plt.legend()
    plt.show()
   


if __name__ == "__main__":
    main()