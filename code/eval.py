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
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import patches

from agent.option_critic import OptionCritic
from env.fourrooms import Fourrooms, LTLFourrooms
from env.breakout import Breakout, LTLBreakout, BreakoutNRA
from env.sapientino import Sapientino, LTLSapientino
from env.cartpole import LTLCartPole
from env.acrobot import LTLAcrobot
from env.taxi import LTLTaxi

from agent.sarsa import Sarsa
from agent.dqn import DQN
from agent.option_critic import OptionCritic
from agent.actor_critic import ActorCritic
from agent.ppo import PPO

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


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
    "LTLacrobot": LTLAcrobot,
    "LTLtaxi": LTLTaxi
}

name = "PPO_LTLtaxi_1"
model = torch.load(f"models/{name}")
cfg = OmegaConf.create(model["hyperparameters"])
cfg.env.render = True

# print(cfg.env.render)
# env = LTLCartPole(**cfg.env)
# print(env.__dict__)
# print(env.render)

# Load env   
##########
if cfg.env.name in envs:
    env = envs[cfg.env.name](**cfg.env)
    # env.env = gym.make("Taxi-v3", render_mode="human")
else:
    env = gym.make(cfg.env.name, render_mode="human")

# env.env = gym.make('CartPole-v1', render_mode="rgb_array")


# Load Agent
############
agent = agents[cfg.agent.name](
        observation_space=env.observation_space, 
        action_space=env.action_space.n,
        args=cfg.agent
)
agent.policy_old.load_state_dict(model['model_params'])
agent.policy.load_state_dict(model['model_params'])

obs, _ = env.reset()

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
frames = []


while not(done or truncated or ep_steps>cfg.agent.max_steps_ep):
    
    # if option_termination:
    #     current_option = np.random.choice(agent.num_options) if np.random.rand() < epsilon else greedy_option

    # ACTION
    # action, logp, entropy, probs = agent.option_critic.get_action(state, current_option)


    action = agent.get_action(state)

    # STEP
    next_obs, reward, done, _, _ = env.step(action)

    state = next_obs
    frames.append(env.render())
    # if reward:
    # print("Ep stesps: ", ep_steps, " reward: ", reward, 
    #         " State: ", env.spec.state, " Angle: ", state[2])

    # img = env.env.render()
    # print(img)

    ep_reward += reward 

    print(cfg.agent.max_steps_ep, ep_steps, reward, ep_reward, done, env.spec.state)
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
env.env.close()
save_frames_as_gif(frames, filename="images/{}.gif".format(cfg.env.name))
