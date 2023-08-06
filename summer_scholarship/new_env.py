import gym
from fourrooms import Fourrooms


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import numpy as np
import argparse
import torch
from copy import deepcopy
from argparse import Namespace

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import Logger

import time


env, is_atari = make_env("ltl_fourrooms", True)

env.reset()
for i in range(10):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    print(action)