import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "code"))

import re
import gymnasium as gym
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
from env.sapientino import Sapientino, LTLSapientino
from env.cartpole import LTLCartPole

from agent.sarsa import Sarsa
from agent.dqn import DQN
from agent.option_critic import OptionCritic
from agent.actor_critic import ActorCritic
from agent.ppo import PPO


agents = {
    "DQN": DQN,
    "OC": OptionCritic,
    "Sarsa": Sarsa,
    "A2C": ActorCritic,
    "PPO": PPO
}

envs = {
    "fourrooms": LTLFourrooms,
    "breakout": LTLBreakout,
    "breakoutNRA": BreakoutNRA,
    "sapientino": LTLSapientino,
    "LTLcartpole": LTLCartPole,
}

name = "PPO_LTLcartpole_1"
model = torch.load(f"models/{name}")
cfg = OmegaConf.create(model["hyperparameters"])
# cfg.env.render = True

# print(cfg.env.render)
# env = LTLCartPole(**cfg.env)
# print(env.__dict__)
# print(env.render)

# Load env   
##########
if cfg.env.name in envs:
    env = envs[cfg.env.name](**cfg.env)
else:
    env = gym.make(cfg.env.name, render_mode="human")

env.env = gym.make('CartPole-v1', render_mode="human")

# agent = OptionCritic(
#         observation_space=env.observation_space, 
#         action_space=env.action_space.n,
#         args=cfg.agent
# )
# agent.option_critic.load_state_dict(model['model_params'])

# Load Agent
############
agent = agents[cfg.agent.name](
        observation_space=env.observation_space, 
        action_space=env.action_space.n,
        args=cfg.agent
)
agent.policy_old.load_state_dict(model['model_params'])
agent.policy.load_state_dict(model['model_params'])

obs = env.reset()

state = obs

# state = agent.option_critic.get_state(obs)
# greedy_option  = agent.option_critic.greedy_option(state)

current_option = 0
curr_op_len = 0
ep_steps = 0
ep_reward = 0
option_termination = True
epsilon = 0
done = False
truncated = False
# max_steps = 200

option_trace = []
spec_trace = []    

# env.env.render("human")


while not(done or truncated or ep_steps>cfg.agent.max_steps_ep):
    
    # if option_termination:
    #     current_option = np.random.choice(agent.num_options) if np.random.rand() < epsilon else greedy_option

    # ACTION
    # action, logp, entropy, probs = agent.option_critic.get_action(state, current_option)

    action = agent.get_action(state)

    # STEP
    next_obs, reward, done, _ = env.step(action)
    state = next_obs

    if reward:
        print(ep_steps, reward, env.spec.state, state[2])

    # img = env.env.render()
    # print(img)

    ep_reward += reward 

    # NEXT STATE
    # state = agent.option_critic.get_state(next_obs)

    # OPTION TERMINATION
    # option_termination, greedy_option = agent.option_critic.predict_option_termination(
    #     state, current_option)
    
    ep_steps += 1
    # curr_op_len += 1
    obs = next_obs
    # option_trace.append(current_option+1)
    # spec_trace.append(env.spec.state)

print(ep_reward)
