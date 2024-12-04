import torch
from torch.utils.data import Dataset 
import numpy as np

"""
Arguments:
data_dir: Absolute path to the dataset directory.
states_filename: Name of states dataset file.
actions_filename: Name of the actions dataset file.
s_transform: Transformation for states.
a_transform: Transformation for actions.

---------------------------------------------------------------------------------
what does it contain ?
states is a numpy array - (num of data points, trajectory_length, 2, 65, 65)
actions is a numpy array - (num of data_points, trajectory_length, 2)
transforms should be image transformations

TODO: check if agent and environment needs the same transformation or different.
"""
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, 
                 states_filename, 
                 actions_filename, 
                 s_transform=None, 
                 a_transform=None,
                 length=None):
        self.states = np.load(f"{data_dir}/{states_filename}", mmap_mode="r")
        self.actions = np.load(f"{data_dir}/{actions_filename}")
        if length is None:
            length = len(self.states)
        
        self.states = self.states[:length]
        self.actions = self.actions[:length]

        self.state_transform = s_transform
        self.action_transform = a_transform
    
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        
        if self.state_transform:
            for i in range(state.shape[0]):
                state[i] = self.state_transform(state[i])
        
        if self.action_transform:
            for i in range(action[i].shape[0]):
                action[i] = self.action_transform(action[i])
        
        return state, action