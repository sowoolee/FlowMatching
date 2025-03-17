import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

import random
import time

import os
from copy import deepcopy
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.models.helpers import apply_conditioning

from isaacgym.torch_utils import quat_rotate_inverse

import onnx
import onnxruntime as ort
import pygame

def load_env(label="gait-conditioned-agility/pretrain-v0/train", headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 20
    Cfg.terrain.num_cols = 20

    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete flat, gap, smooth plain, ]
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    # discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
    Cfg.terrain.slope_treshold = 0.
    Cfg.terrain.max_platform_height = 0.
    Cfg.terrain.difficulty_scale = 1.
    print("default slope : ", Cfg.terrain.slope_treshold)
    print("difficulty : ", Cfg.terrain.difficulty_scale)
    print("platform height : ", Cfg.terrain.max_platform_height)

    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 1
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    return env


def batch_to_device(batch, device):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def import_trainer(path):
    import flow.utils as utils
    from config.train_config import Config

    Config.device = 'cuda:0'

    basepath = '/home/hubolab/workspace/FM/weights/flow/'
    loadpath = os.path.join(basepath, path, 'checkpoint', 'state_100000.pt')
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        action_scale=Config.action_scale
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        device=Config.device,
    )

    flow_config = utils.Config(
        Config.flow,
        savepath='flow_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        record_freq=Config.record_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    flow_model = flow_config(model)
    trainer = trainer_config(flow_model, dataset, renderer)

    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    return trainer


def test():
    env = load_env()

    # import diffusion model
    trainer = import_trainer('4gait/unet')
    dataset = trainer.dataset
    device = trainer.device
    renderer = trainer.renderer

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5

    # start testing
    t = 0
    env.reset()
    total_steps = 250
    state_traj = []
    inference_time = 0

    measured_x_vels = np.zeros(total_steps)
    measured_y_vels = np.zeros(total_steps)
    planned_x_vels = np.zeros(total_steps)
    planned_y_vels = np.zeros(total_steps)
    target_x_vels = np.ones(total_steps) * v_x

    while t < total_steps:
        if t < total_steps / 2:
            gait_num = 1
        else:
            gait_num = 2
        returns = to_device(torch.Tensor([[gait_num, v_x, 0,0.5] for i in range(num_envs)]), device)

        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        s_t = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)
        state_traj.append(s_t)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.,0.]]), obs[:,2:]], axis=-1)

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        start = time.time()
        samples = trainer.ema_model.inference(conditions, returns)
        end = time.time()
        inference_time += (end - start)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        # if t==30:
        #     planned_linvel = to_np(
        #         quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))
        #     )
        #     for i in range(planned_linvel.shape[0]):
        #         planned_x_vels[i] = planned_linvel[i][0]
        #         planned_y_vels[i] = planned_linvel[i][1]

        with torch.no_grad():
            action = trainer.ema_model.inv_model(obs_comb)
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        measured_x_vels[t] = env.base_lin_vel[0, 0]
        measured_y_vels[t] = env.base_lin_vel[0, 1]

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

    # target_vel = np.array([v_x, 0])
    # planned_xy = planned_linvel[:,:2]
    # measured_xy = np.stack([measured_x_vels, measured_y_vels], axis=1)
    # diff = measured_xy - target_vel # planned_xy - target_vel
    # velocity_norms = np.linalg.norm(diff, axis=1)  # (56,)
    # # print("Velocity differences (norm):", velocity_norms)
    # print("Average Velocity Tracking RMS Error: ", np.mean(velocity_norms))
    # print("Average Inference Time: ", inference_time / total_steps, "s")
    #
    # # play recorded trajectory
    # recorded_traj = np.stack(state_traj, axis=1)
    # renderer.composite3(recorded_traj, 'test', 'trot')
    #
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_x_vels, color='black', linestyle="-", label="Measured_x")
    # axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_y_vels, color='black', linestyle="-", label="Measured_y")
    # axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")
    #
    # axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_x_vels, color='black', linestyle="-", label="Measured_x")
    # axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_y_vels, color='black', linestyle="-", label="Measured_y")
    # axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[1].legend()
    # axs[1].set_title("Planned Forward Linear Velocity")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Velocity (m/s)")
    #
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    test()
