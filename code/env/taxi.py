import gym
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from specs import Specification


class LTLTaxi:

    def __init__(self, name, render, spec) -> None:
        
        self.name = name
        if render:
            self.env = gym.make('Taxi-v3', render_mode="rgb_array")
        else:
            self.env = gym.make('Taxi-v3')

        self.spec = Specification(**spec)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(4+len(self.spec),))
        self.action_space = self.env.action_space

        self.colors = [
            [0, 0], # Red
            [4, 0], # Green
            [0, 3], # Yellow
            [3, 3], # Blue
        ]

    def reset(self):

        env_state, _ = self.env.reset()
        env_state = np.array(list(self.env.decode(env_state)))

        spec_state = self.spec.reset()

        prod_state = np.concatenate([env_state, spec_state], 0)

        return prod_state, None
    
    def label_state(self, state, action):
        # Actions:
        # 0: move south
        # 1: move north
        # 2: move east
        # 3: move west
        # 4: pickup passenger
        # 5: drop off passenger

        pos = list(state[:2])

        # print(pos, pick_pos, pos==pick_pos)
        variables = {
            "p": action==4 and state[2]<4 and pos==list(self.colors[state[2]]),
            "d": action==5 and pos==list(self.colors[state[3]]),     
            "failure": (action==4 and (state[2]==4 or pos!=list(self.colors[state[2]])))\
                  or (action==5 and pos!=list(self.colors[state[3]])),
        }

        return variables
    
    def step(self, action):

        # Fourroom State
        env_state, env_reward, env_done, _, _ = self.env.step(action)
        env_state = np.array(list(self.env.decode(env_state)))

        # Spec State
        variables = self.label_state(env_state, action)
        variables["alive"] = not env_done
        spec_state, spec_reward, spec_done = self.spec.step(**variables)

        # Prod State
        next_prod_state = np.concatenate([env_state, spec_state], 0)

        # Total Reward
        reward = spec_reward if spec_reward else -1
        if variables["failure"]:
            reward = -10

        if variables["d"]:
            spec_done = True

        # print(next_prod_state, variables, reward)

        done = spec_done or env_done

        return next_prod_state, reward, done, False, None
    
    def render(self):

        # Return the screen render
        return self.env.render()
    