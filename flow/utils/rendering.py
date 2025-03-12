import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym

import warnings
import pdb

import raisimpy as raisim
import math
import time
from scipy.spatial.transform import Rotation

from .arrays import to_np
from ml_logger import logger

from datetime import datetime

from diffuser.datasets.d4rl import load_environment


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class RaisimRenderer:
    '''
        raisim renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            self.env = env
        else:
            self.env = env

        raisim.World.setLicenseFile("/home/hubolab/.raisim/activation.raisim")
        world_ = raisim.World()
        ground = world_.addGround()

        anymal_ = world_.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../environments/assets/go1/go1.urdf")

        self.world = world_
        self.anymal = anymal_
        self.dt = 0.02
        self.worldTime = 0

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256):
        state = observation
        quat = [state[6], state[3], state[4], state[5]]
        dof_pos = state[-24:-12]

        # convert from isaac to raisim convention
        dof_pos = [*dof_pos[3:6], *dof_pos[0:3], *dof_pos[9:12], *dof_pos[6:9]]

        gc = [  *state[0:3],
                *quat,
                *dof_pos  ]

        self.world.setWorldTime(self.worldTime)
        self.anymal.setGeneralizedCoordinate(gc)
        time.sleep(0.1)

        data = np.zeros((*dim, 3), np.uint8)
        self.worldTime += self.dt
        return data

    def _renders(self, observations, **kwargs):
        for observation in observations:
            self.render(observation, **kwargs)
        return None

    def renders(self, samples, **kwargs):
        self._renders(samples, **kwargs)
        return None

    def composite(self, paths, dim=(1024, 256), **kwargs):
        self.worldTime = 0

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        server.focusOn(self.anymal)

        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            self.renders(to_np(path), dim=dim, **kwargs)

        server.killServer()

        return images

    def composite3(self, paths, title, name, dim=(1024, 256), **kwargs):
        self.worldTime = 0

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        current_date = datetime.now().strftime("%m%d")
        directory_path = os.path.expanduser(f'~/workspace/raisim/raisimLib/raisimUnity/linux/Screenshot/{current_date}/{title}')
        os.makedirs(directory_path, exist_ok=True)
        video_path = current_date + "/" + title + "/" + name + ".mp4"
        server.startRecordingVideo(video_path)
        server.focusOn(self.anymal)
        time.sleep(1)

        images = []
        for path in paths:
            path = atmost_2d(path)
            self.renders(to_np(path), dim=dim, **kwargs)

        server.stopRecordingVideo()
        server.killServer()

        return images