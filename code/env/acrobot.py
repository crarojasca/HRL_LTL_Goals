import gym
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from specs import Specification


class LTLAcrobot:

    def __init__(self, name, render, spec) -> None:
        
        self.name = name
        if render:
            self.env = gym.make('Taxi-v3', render_mode="rgb_array")
        else:
            self.env = gym.make('Taxi-v3')

        self.spec = Specification(**spec)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(self.env.observation_space.shape[0]+len(self.spec),))
        self.action_space = self.env.action_space

    def reset(self):

        env_state, _ = self.env.reset()
        spec_state = self.spec.reset()

        prod_state = np.concatenate([env_state, spec_state], 0)

        return prod_state, None
    
    def label_state(self, state):
        # State:
        # - Cosine of theta1
        # - Sine of theta1
        # - Cosine of theta2
        # - Sine of theta2
        # - Angular velocity of theta1
        # - Angular velocity of theta2

        angle1 = 0.5
        angle2 = 0.15

        variables = {
            "a": math.asin(state[1]) > angle1,
            "b": math.asin(state[3]) < abs(angle2)
        }

        return variables
    
    def step(self, action):

        # Fourroom State
        env_state, env_reward, env_done, _, _  = self.env.step(action)

        # Spec State
        variables = self.label_state(env_state)
        variables["alive"] = not env_done
        spec_state, spec_reward, spec_done = self.spec.step(**variables)

        # print(variables, self.spec.state)

        # Prod State
        next_prod_state = np.concatenate([env_state, spec_state], 0)

        # Total Reward
        reward = env_reward + spec_reward

        return next_prod_state, spec_reward, env_done, False, None
    
    def render(self):

        # Return the screen render
        return self.env.render()
    