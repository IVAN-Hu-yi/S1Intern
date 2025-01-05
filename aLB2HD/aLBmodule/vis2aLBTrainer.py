import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aLBCells import ALBcells
from visionModule.visualCells import visualCells
from utilities.parameters import DefaultParameters
from tqdm import tqdm
import numpy as np
from HD.HDRing import HDattractor
from scipy.io import loadmat
from utilities.AngularDiff import AngularDiff


class vis2aLBPathway():

    def __init__(self, 
                 params:DefaultParameters, 
                 duration, 
                 N_env, 
                 N_cue, 
                 Cue_Init, 
                 Strength_Init, 
                 firingrate_cirterion, 
                 HDdata,
                 wandering, 
                 wanderingTime=None):

        if wandering == 1 and wanderingTime is None:
            raise ValueError("wanderingTime must be specified when the agent is allowed to wander")
        
        self.params = params
        self.params.time = duration
        self.params.N_env = N_env
        self.params.N_cue = N_cue
        self.params.N_input = self.params.N_cue * self.params.N_bin
        self.params.Cue_Init = Cue_Init
        self.params.Strength_Init = Strength_Init
        self.params.firingrate_cirterion = firingrate_cirterion

        # wandering
        self.params.wandering = wandering
        self.params.wanderingTime = wanderingTime

        # global variables
        self.params.Cue_global = self.params.Cue_Init.copy()
        self.params.Strength_global = self.params.Strength_Init.copy()

        # modules initialization
        self.HDModule = HDattractor()
        self.VisualModule = visualCells(self.params)
        self.ALBModule = ALBcells(self.params.N_bin, self.params)

        # HD data
        self.HDModule.set_HD_trajectory(HDdata) # set the HD trajectory
        print(f'HD trajectory shape: {self.HDModule.trajectory.shape}')

        print('aLBTrainer initialized')

        # time handling
        self.training_idicies = np.arange(0, self.params.time+self.params.dt_exp, self.params.dt_exp)
        self.T_len = len(self.training_idicies)
        self.params.time_CueShifting = np.arange(0, self.params.time + (self.params.time / self.params.N_env), (self.params.time))
        self.stop_learning_time = self.params.time

        print(f'nb of timepoints to shift:{len(self.params.time_CueShifting)}')


    def train(self, store=False, interval=50, mode='all', last_nb:int=10000):
        '''
        Training the aLB module to obtain the abstract Landmark-based representation by updating vis2aLB weights with lateral inhibition
        '''
        print('Start training the aLB module')

        previous_env = 0

        # data collection
        if store: 
            if mode == 'all':
                recordTimePoints = int(self.T_len/interval)
                self.vis2ALBweights = np.zeros((recordTimePoints, self.params.N_abstract, self.params.N_input))
                # firing rates
                self.aLB_fr = np.zeros((recordTimePoints, self.params.N_abstract))

            elif mode == 'last':
                start_idx = self.T_len - last_nb
                self.vis2ALBweights = np.zeros((last_nb, self.params.N_abstract, self.params.N_input))
                self.aLB_fr = np.zeros((last_nb, self.params.N_abstract))

        for i in tqdm(range(self.T_len), desc='Training'):
            
            # time handling
            curr_time_global = 0 + (i+1) * self.params.dt

            # env handling
            if self.params.wandering:
                present_env = np.round(curr_time_global / self.params.wanderingTime, 0) % self.params.N_env + 1 
            
            else:
                if previous_env < self.params.N_env:
                   present_env =  np.argmax(self.params.time_CueShifting >= curr_time_global) if np.any(self.params.time_CueShifting >= curr_time_global) else None

            if present_env > previous_env:
                previous_env = present_env

            # store previous values
            self.VisualModule.store_as_previous()
            self.ALBModule.store_as_previous()

            # update the visual module
            self.VisualModule.update_F_visual(self.HDModule.trajectory, present_env-1, i)

            # update the aLB module
            self.ALBModule.update_U_arep(self.VisualModule, i)
            self.ALBModule.update_F_arep()

            # lr update
            self.ALBModule.update_lr(i, curr_time_global, self.stop_learning_time)

            # weight update
            self.ALBModule.vis2aLB_weight_update(self.VisualModule, i)
            
            if store:
                if mode == 'all':
                    if store and i % interval == 0:
                        self.vis2ALBweights[i // interval] = self.ALBModule.W_vis2aLB.copy()
                        self.aLB_fr[i // interval] = self.ALBModule.F_arep.copy()
                
                if mode == 'last':
                    if i >= start_idx:
                        idx = i - start_idx
                        self.vis2ALBweights[idx] = self.ALBModule.W_vis2aLB.copy()
                        self.aLB_fr[idx] = self.ALBModule.F_arep.copy()
        print('Training completed')

    def _clear_data(self):

        # reset ALB cells
        self.ALBModule.F_arep = np.zeros(self.ALBModule.F_arep.shape)
        self.ALBModule.U_arep = np.zeros(self.ALBModule.U_arep.shape)
        self.ALBModule.F_arep_p = np.zeros(self.ALBModule.F_arep_p.shape)
        self.ALBModule.U_arep_p = np.zeros(self.ALBModule.U_arep_p.shape)

        # reset visual cells
        self.VisualModule.F_visual = np.zeros(self.VisualModule.F_visual.shape)
        self.VisualModule.F_visal_p = np.zeros(self.VisualModule.F_visal_p.shape)

        # clear timeline storage
        self.vis2ALBweights = None
        self.aLB_fr = None
        
    def test(self, store=False):
        '''
        Test the aLB module by updating vis2aLB weights with lateral inhibition
        
        Parameters
        ----------
        filepath : str
            path to store the data
        store : bool, optional
            check if store the data, by default False
        '''

        print('Start testing the aLB module')

        ## parameters
        # time handling
        start_time = 0
        end_time = 60
        test_idicies = np.arange(start_time, end_time+self.params.dt, self.params.dt)
        cycle = 10
        vlcty = 360 * cycle / (end_time - start_time)
        self.testing_T_len = len(test_idicies)

        # Angle handling
        bin_width_multiplier = 1
        bar_angle_gap = bin_width_multiplier * self.params.angle_gap # bin width
        bar_angle = np.arange(-180, 180+bar_angle_gap, bar_angle_gap)

        # data collection
        recorded_aLB_fr = np.zeros((self.testing_T_len, self.params.N_abstract))

        # test HD trajectory
        self.test_HD_trajectory = np.zeros(self.testing_T_len)

        for i in tqdm(range(self.testing_T_len), desc='Testing'):
            # time handling
            curr_time_global = 0 + (i+1) * self.params.dt

            # env handling
            present_env = 0

            # HD trajectory update
            self.test_HD_trajectory[i] = AngularDiff(self.test_HD_trajectory[i-1] + vlcty * self.params.dt, 0) 
            # store previous values
            self.VisualModule.store_as_previous()
            self.ALBModule.store_as_previous()

            # update the visual module
            self.VisualModule.update_F_visual(self.HDModule.trajectory, present_env-1, i)

            # update the aLB module
            self.ALBModule.update_U_arep(self.VisualModule, i)
            self.ALBModule.update_F_arep()

            # store the firing rate
            if store:
                recorded_aLB_fr[i] = self.ALBModule.F_arep.copy()

    def checkfilepath_or_create(self, path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

    def save_training_data(self, filepath, store=True):

        print('Start saving the training data') 
        self.checkfilepath_or_create(filepath)

        # store visual
        rp = filepath + '/visualComponent'
        self.checkfilepath_or_create(rp) 
        np.save(rp+'/F_visual.npy', self.VisualModule.F_visual)
        np.save(rp+'/F_featural_attention.npy', self.VisualModule.Inputs.F_visual_feature)
        
        # store aLB cells
        rp = filepath + '/aLB'
        self.checkfilepath_or_create(rp)
        np.save(rp+'/vis2aLBWeights.npy', self.ALBModule.W_vis2aLB)
        np.save(rp+'/aLB_fr.npy', self.ALBModule.F_arep)

        # store timeline changes
        if store:
            rp = filepath + '/timeline'
            self.checkfilepath_or_create(rp)
            np.save(rp+'/vis2aLBWeights.npy', self.vis2ALBweights)
            np.save(rp+'/aLB_fr.npy', self.aLB_fr)
            # corresponding HD trajectory
            np.save(rp+'/HDtrajectory.npy', self.HDModule.trajectory)

    def save_testing_data(self, filepath):

        print('Start saving the testing data')
        self.checkfilepath_or_create(path=filepath)

        # store aLB cells
        rp = filepath + '/aLB'
        self.checkfilepath_or_create(rp)
        np.save(rp+'/aLB_fr.npy', self.aLB_fr)

        # store HD trajectory
        rp = filepath + '/HD'
        self.checkfilepath_or_create(rp)
        np.save(rp+'/HDtrajectory.npy', self.test_HD_trajectory)

        print('Data saved')


def main():

    params = DefaultParameters()

    duration = 60*20
    N_env = 1
    N_cue = 2
    Cue_Init = np.ones((N_env, N_cue))*360 - 360
    Strength_Init = np.ones(( N_env, N_cue ))
    firingrate_cirterion = 0.5
    wandering = 0
    wanderingTime = 10

    # data storage
    store = True
    interval = 10 # interval of storing data
    mode = 'last' # 'all' or 'last' mode of storing data
    last_nb = 10000 # number of last timepoints to store

    filename = 'aLBcells-for-HDsystem/RealData_CIRC_Manson/RealData_CIRC_Manson1.mat'
    HDdata = loadmat(filename) 

    vis2aLB = vis2aLBPathway(params, duration, N_env, N_cue, Cue_Init, Strength_Init, firingrate_cirterion, HDdata, wandering, wanderingTime )

    vis2aLB.train(store=False, interval=interval, mode=mode, last_nb=last_nb)
    
    vis2aLB.save_training_data('data/train', store=False)

    vis2aLB.test(store=True)

    vis2aLB.save_testing_data(filepath='data/test')

if __name__ == '__main__':
    main() 