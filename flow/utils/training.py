import isaacgym
assert isaacgym

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import einops
import pdb
import diffuser
from copy import deepcopy

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch, Timer
from ml_logger import logger

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from adamp import AdamP
import random

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
            self,
            flow_model,
            dataset,
            renderer,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=1000,
            save_freq=1000,
            eval_freq=1000,
            record_freq=50000,
            label_freq=100000,
            save_parallel=False,
            n_reference=4,
            bucket=None,
            train_device='cuda',
            save_checkpoints=False,
    ):
        super().__init__()
        self.model = flow_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.record_freq = record_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_lr, weight_decay=1e-3)
        # self.optimizer = AdamP(self.model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=1e-2)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        self.env = None
        self.action_scale = dataset.action_scale

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        # self.load()

        # state_dict = torch.load(loadpath, map_location=Config.device)
        # self.ema_model.load_state_dict(state_dict['model'])

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        writer = SummaryWriter()
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                if step % 2000 == 0:
                    x_targ, x_pred, a_targ, a_pred = self.model.loss_debug(*batch)

                    normed_loss = (to_np(x_targ) - to_np(x_pred)) ** 2
                    normed_loss = np.mean(normed_loss, axis=(0,1))
                    targ_unnormed = self.dataset.normalizer.unnormalize(to_np(x_targ), 'observations')
                    pred_unnormed = self.dataset.normalizer.unnormalize(to_np(x_pred), 'observations')
                    org_loss = (targ_unnormed - pred_unnormed) ** 2
                    org_loss = np.mean(org_loss, axis=(0,1))

                    a_unnormed = to_np(a_targ) * self.action_scale
                    a_pred_unnormed = to_np(a_pred) * self.action_scale
                    unnormed_inv_loss = (a_pred_unnormed - a_unnormed) ** 2
                    unnormed_inv_loss = np.mean(unnormed_inv_loss, axis=(0,1))

                    writer.add_scalar("loss/unnormed inv loss", unnormed_inv_loss, step)

                    writer.add_scalar("error/base_pos[m]", np.sqrt(np.sum(normed_loss[0:2])), step)
                    writer.add_scalar("error/base_ori[quat]", np.sqrt(np.sum(org_loss[3:7])), step)
                    writer.add_scalar("error/base_lin_vel[m/s]", np.sqrt(np.sum(org_loss[7:10])), step)
                    writer.add_scalar("error/base_ang_vel[rad/s]", np.sqrt(np.sum(org_loss[10:13])), step)
                    writer.add_scalar("error/joint_pos[rad]", np.sqrt(np.sum(org_loss[-24:-12])), step)
                    writer.add_scalar("error/joint_vel[rad/s]", np.sqrt(np.sum(org_loss[-12:])), step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                print('----------------------------------------------------')
                print('{:>6}th iteration'.format(self.step))
                print('{:<40} {:>8.4f}'.format("Loss: ", loss))
                print('{:<40} {:>8.4f}'.format("Time elapsed in this iteration: ", timer()))
                print('----------------------------------------------------\n')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.step and self.step % self.record_freq == 0:
                self.record_samples()

            self.step += 1
        writer.close()

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        trajectories = to_np(batch.trajectories)

        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        scaled_xy = normed_observations[:, :, 0:2]
        observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

        self.renderer.composite(observations)

    def record_samples(self, batch_size=1, n_samples=1):
        '''
            renders samples from model
        '''
        for i in range(batch_size):
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # nominal_gc = [[25, 25, 0.3, 0, 0, 0, 1, 0,0,0, 0,0,0, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5, 0,0,0,0,0,0,0,0,0,0,0,0]]
            # nominal_gc = self.dataset.normalizer.normalize(to_np(nominal_gc), 'observations')
            # conditions = {0: torch.Tensor(nominal_gc).to(self.device)}

            commands = [[1, 1.5, 0, 0]] #, [2, 1.5, 0, 0], [3, 1.5, 0, 0], [0, 1.5, 0, 0]]
            # commands = [[1, 0.8, 0, 0], [1, -0.8, 0, 0], [1, 0, 0.4, 0], [1, 0, 0, 0.8]]
            # commands = [[2, 0.8, 0, 0], [2, -0.8, 0, 0], [2, 0, 0.4, 0], [2, 0, 0, 0.8]]
            # commands = [[3, 0.8, 0, 0], [3, -0.8, 0, 0], [3, 0, 0.4, 0], [3, 0, 0, 0.8]]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            returns = to_device(torch.Tensor([commands[i] for i in range(n_samples)]), self.device)

            samples = self.ema_model.inference(conditions, returns=returns)
            # samples = self.ema_model.inference_1step(conditions, returns=returns)
            samples = to_np(samples)

            normed_observations = samples[:, :, :]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            scaled_xy = normed_observations[:, :, 0:2]
            observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

            title = 'train'
            name = str(self.step)
            self.renderer.composite3(observations, title, name)
