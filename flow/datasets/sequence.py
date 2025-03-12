from collections import namedtuple
import numpy as np
import torch
import time

from .rawdata import read_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
                 normalizer='LimitsNormalizer', max_path_length=1000,
                 max_n_episodes=100000, termination_penalty=0, use_padding=True, include_returns=False,
                 action_scale=1.0):
        self.env = env
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = read_dataset()

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        print("measuring time from now")
        start_time = time.time()
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        end_time = time.time()
        print("elapsed time: ", end_time - start_time, "sec")

        print(fields)

        self.action_scale = action_scale

    def normalize(self, keys=['observations']): #, 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        unnormed_obs = self.fields.observations[path_ind, start:end]
        unnormed_xy = unnormed_obs[:,0:2]
        normed_obs = self.fields.normed_observations[path_ind, start:end]
        normed_other = normed_obs[:,2:]
        # scale manually
        unnormed_xy = (unnormed_xy - unnormed_xy[0:1,0:2])
        observations = np.concatenate([unnormed_xy, normed_other], axis=-1)

        actions = self.fields.actions[path_ind, start:end] / self.action_scale

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            ################## reward gait version #####################
            rewards = self.fields.rewards[path_ind, start:end]
            returns = rewards[0]
            ############################################################
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch