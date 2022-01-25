"""
Dataset for RedNet-RNN training

Based off https://github.com/JindongJiang/RedNet
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


train_envs = [
    '17DRP5sb8fy', '759xd9YjKW5', 'D7N2EKCX4Sj', 'JeFG25nYj2p', 'Uxmj2M2itWa', 'XcA2TqTSSAj', 'cV4RVeZvu5T', 'kEZ7cmS4wCh', 'r47D5H71a5s',
    'ur6pFq6Qu1A', '1LXtFkjw3qL', '7y3sRwLe3Va', 'E9uDoFAP3SH', 'PX4nDJXEHrG', 'V2XKFyX4ASd', 'YmJkqBEsHnH', 'dhjEzFoUFzH', 'mJXqzFtmKg4',
    'rPc6DW4iMge', 'vyrNrziPKCB', '1pXnuDYAj8r', '82sE5b5pLXE', 'EDJbREhghzL', 'Pm6F8kyY3z2', 'VFuaQ6m2Qom', 'ZMojNkEp431', 'e9zR4mvMWw7',
    'p5wJjkQkbXX', 's8pcmisQ38h', '29hnd4uzFmX', '8WUmhLawc2A', 'GdvgFV5R1Z5', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d', 'gZ6f7yhEvPG',
    'pRbA3pwrgk9', 'sKLMLpTHeUy', '5LpN3gDmAk7', 'B6ByNegPMKs', 'HxpKQynjfin', 'S9hNv5qa7GM', 'VVfe2KiqLaN', 'ac26ZMwG7aT', 'i5noydFURQK',
]

test_envs = [
    'qoiz87JEwZ2', 'sT4fr6TAbpF', '5q7pvUzZiYa', 'D7G3Y4RVNrH', 'JF19kD82Mey', 'ULsKaCPVFJR', 'Vvot9Ly1tCj', 'b8cTxDM8gDG', 'jh4fc5c5qoQ',
    'r1Q1Z4BcV1o', 'uNb9QFRL6hY'
]

ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
ACTIONS_MAP = {act: i for i, act in enumerate(ACTIONS)}

keys = ['rgb', 'depth', 'semantic',]# 'objectgoal', 'compass', 'gps']

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, seq_len=500, transform=None, phase_train=True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.phase_train = phase_train

        self.trajectories = []
        self.envs = train_envs if self.phase_train else test_envs
        for env in self.envs:
            for traj in os.listdir(os.path.join(data_dir, env)):
                self.trajectories.append(os.path.join(data_dir, env, traj))
        self.num_trajectories = len(self.trajectories)
    
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, index):
        dir = self.trajectories[index]

        observations = {k: [] for k in keys}
        for t in range(self.seq_len):
            obs = np.load(os.path.join(dir, '{}.npy'.format(t)), allow_pickle=True).item()
            for k in keys:
                observations[k].append(obs[k])
        
        sample = {k: np.array(v) for k, v in observations.items()}
        
        actions = np.load(os.path.join(dir, 'actions.npy'.format(t)), allow_pickle=True)
        sample_actions = [0]
        for t in range(self.seq_len-1):
            sample_actions.append(ACTIONS_MAP[actions[t]])
        sample['actions'] = np.array(sample_actions)

        if self.transform:
            sample = self.transform(sample)

        return sample

