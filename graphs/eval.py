import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "code"))

import re
import json
import hydra
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

from agent.option_critic import OptionCritic
from env.fourrooms import Fourrooms, LTLFourrooms
from env.breakout import Breakout, LTLBreakout, BreakoutNRA

name = "OC_breakout_1_8opt"
model = torch.load(f"models/{name}")
cfg = OmegaConf.create(model["hyperparameters"])
cfg.env.render = True
env = LTLBreakout(**cfg.env)
agent = OptionCritic(
        observation_space=env.observation_space, 
        action_space=env.action_space.n,
        args=cfg.agent
)
agent.option_critic.load_state_dict(model['model_params'])

obs, info = env.reset()
state = agent.option_critic.get_state(obs)
greedy_option  = agent.option_critic.greedy_option(state)

current_option = 0
curr_op_len = 0
ep_steps = 0
ep_reward = 0
option_termination = True
epsilon = 0
done = False
truncated = False
max_steps = 200

option_trace = []
spec_trace = []    



while not(done or truncated or ep_steps>max_steps):
    
    if option_termination:
        current_option = np.random.choice(agent.num_options) if np.random.rand() < epsilon else greedy_option

    # ACTION
    action, logp, entropy, probs = agent.option_critic.get_action(state, current_option)

    # STEP
    next_obs, reward, done, truncated, _ = env.step(action)
    ep_reward += reward 

    # NEXT STATE
    state = agent.option_critic.get_state(next_obs)

    # OPTION TERMINATION
    option_termination, greedy_option = agent.option_critic.predict_option_termination(
        state, current_option)
    
    ep_steps += 1
    curr_op_len += 1
    obs = next_obs
    option_trace.append(current_option+1)
    spec_trace.append(env.spec.state)
