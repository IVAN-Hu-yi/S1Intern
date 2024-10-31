import numpy as np
from typing import Literal

class DefaultParameters:
    def __init__(self, mode: Literal['default', 'inhibition', 'no_inhibition'] = 'default'):
        # Timing
        self.time = 60 * 20  # Duration time in seconds (Experimental Duration: 20 mins)
        self.beginning = 60 * 0  # Initial time point in seconds
        self.stop_learning_time = self.beginning + self.time  # End time for learning

        # Time interval (seconds)
        self.dt_exp = 0.02
        self.Operation_Noatun = False
        self.dt = 0.002 if self.Operation_Noatun else self.dt_exp
        
        self.Time = np.arange(0, self.time + self.dt_exp, self.dt_exp)  # Time points
        self.T_len = len(self.Time)  # Number of time points

        # Angular interval (degrees)
        self.angle_gap = 1  # Please ensure that the gap of angle can divide 180!
        self.Angle = np.arange(-180, 180, self.angle_gap)  # Preferred direction
        

        # Number of units with distinct preferred directions
        self.N_bin = len(self.Angle)

        # Number of subsequent environments
        self.N_env = 1

        # Number of distinguishable cues
        self.N_cue = 2

        # Number of visual input layer units
        self.N_input = self.N_cue * self.N_bin

        # Number of abstract layer units
        self.N_abstract = 360

        # Angular position (degrees) of the center of cues (centralized at initial HD trajectory)
        self.Cue_Init = np.ones((self.N_env, self.N_cue)) * 360 - 360
        self.Strength_Init = np.ones((self.N_env, self.N_cue))
        self.Cue_global = self.Cue_Init.copy()
        self.Strength_global = self.Strength_Init.copy()

        # Interinhibition matrix of abstract representation
        if mode == 'default':
            self.Inhibition_U_arep = (np.ones((self.N_abstract, self.N_abstract)) - np.eye(self.N_abstract)) / np.sqrt(self.N_abstract - 1)  # Lateral inhibition
        # Uncomment the appropriate line for different inhibition configurations
        elif mode == 'inhibition':
            self.Inhibition_U_arep = np.ones((self.N_abstract, self.N_abstract)) / np.sqrt(self.N_abstract)  # Global inhibition
        elif mode == 'no_inhibition':
            self.Inhibition_U_arep = np.zeros((self.N_abstract, self.N_abstract))

        # Artificial HD velocity (deg/s)
        self.vlcty_tuning = 100

        # Time point (s) for shifting cues
        self.time_CueShifting = np.arange(self.beginning + (self.time / self.N_env), self.time + (self.time / self.N_env), (self.time / self.N_env))

        # Membrane Potential Time (s) & Decay Rate
        self.time_constant_arep = 0.02
        self.decay_rate_arep = self.dt / self.time_constant_arep
        self.time_constant_dRSC = 0.02
        self.decay_rate_dRSC = self.dt / self.time_constant_dRSC
        self.time_constant_gRSC = 0.02
        self.decay_rate_gRSC = self.dt / self.time_constant_gRSC
        self.time_constant_HD = 0.02
        self.decay_rate_HD = self.dt / self.time_constant_HD
        
        self.time_constant_vSTM = 0.5
        self.decay_rate_vSTM = self.dt / self.time_constant_vSTM;
        self.time_constant_interval = 1;
        self.decay_rate_interval = self.dt / self.time_constant_interval;

        # activation function
        self.alpha_arep = 0;
        self.beta_arep = 0.05;
        self.gamma_arep = 0;
        self.alpha_dRSC = 0;
        self.beta_dRSC = 0.04;
        self.gamma_dRSC = 0;
        self.alpha_gRSC = 0;
        self.beta_gRSC = 0.04;
        self.gamma_gRSC = 0;
        self.alpha_HD = 20;
        self.beta_HD = 0.08;
        self.gamma_HD = 0;

        # learning rate 
        self.lr_initial_rate_visual = 1e-3;
        self.lr_decay_rate_visual = 0; # Hz
        self.dr_weight_visual = 0; 
        self.lr_initial_rate_arep2dRSC = 1e-4;
        self.lr_decay_rate_arep2dRSC = 0; # Hz
        self.lr_initial_rate_g2dRSC = 5e-5; 
        self.lr_decay_rate_g2dRSC = 0; # Hz

        self.lr_initial_rate_arep2dRSC_slow = self.lr_initial_rate_arep2dRSC; 
        self.lr_decay_rate_arep2dRSC_slow = self.lr_decay_rate_arep2dRSC; # Hz
        self.lr_initial_rate_g2dRSC_slow = self.lr_initial_rate_g2dRSC; 
        self.lr_decay_rate_g2dRSC_slow = self.lr_decay_rate_g2dRSC; # Hz

        # Precision settings for visual input
        self.precision_visual1 = 20
        self.precision_visual2 = 80
        self.precision_HD = 15
        self.precision_sub = 1
        self.precision_visualfield = 1

        # angular length of proximal cue
        self.proximal_length = 180

        # Gain factors
        self.inhibition_U_arep = 500
        self.Uv_gain_factor = 2
        self.U_arep2dRSC_gain_factor = 50
        self.U_g2dRSC_gain_factor = 5
        self.U_dRSC2dRSC_gain_factor = 50
        self.U_gRSC2gRSC_gain_factor = 50
        self.U_HD2gRSC_gain_factor = 50
        self.U_HD_gain_factor = 1
        self.U_dRSC2HD_gain_factor = 0.1
        self.U_dRSC2HD_i_gain_factor = 2
        self.U_gRSC2HD_gain_factor = 0
        self.Wv_weight_scale = 10
        self.W_arep2dRSC_weight_scale = 1
        self.W_g2dRSC_weight_scale = 1
        self.W_dRSC2HD_weight_scale = 1

        # Maximum of visual firing rate
        self.Fv_max_factor = 1.0

        # Minimum firing rate (scaled) for enabling to encode
        self.firingrate_criterion = 0.5  # Will later plot all cells if no one exceeds the threshold

        # Visual noise intensity
        self.visual_noise_intensity = 0

        # Theta wave settings for encoding (feedforward: ff) & retrieving (feedback: fb)
        self.theta_intensity_ff = 0  # [0, 1]
        self.theta_intensity_fb = 0
        self.theta_frequency_ff = 5  # Hz
        self.theta_frequency_fb = 5
        self.theta_phase_ff = 0  # Degrees
        self.theta_phase_fb = 180

        # Total resource of visual featural attention to abstract layer
        self.total_featural_attention = self.N_cue

        # Anticipatory time interval (seconds)
        self.rho = 0  # Originally 0.025

        # Regularization for weight
        self._lambda = 1

        # Stable weights with noise or bias
        self.Operation_Ratatoskr = 0
        self.weight_bias = self.Operation_Ratatoskr * 0
        self.weight_noise_scale = self.Operation_Ratatoskr * 0.0005
        self.asyweightstrength_noise_scale = self.Operation_Ratatoskr * 0  # Noise of asymmetric weight strength

        # Parameters for initial HD activity level
        self.angluarrepresentation_phase = 0  # Degrees
        self.noise_scale = 0

        # Plotting settings
        self.row = 2
        self.column = 3
        self.Black_White = [0.9, 0.6, 0.3, 0.0]
        self.Color = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]
        self.OrdinalNum = ["st", "nd", "rd", "th"]
        self.time_duration_around = 10  # Time duration around changing time (seconds)
        self.max_selection = 4  # Preparing for showing abstract representation


