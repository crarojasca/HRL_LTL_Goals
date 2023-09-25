import gym
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from specs import Specification


class LTLCartPole:

    def __init__(self, name) -> None:
        
        self.name = name
        self.env = gym.make('CartPole-v1')
        self.spec = Specification(
            formula="G ((a | b) & G ( (a -> (Fb | alive)) & (b -> (Fa| !alive)) ) & G !(a & b))")
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(self.env.observation_space.shape[0]+len(self.spec),))
        self.action_space = self.env.action_space

    def reset(self):

        env_state, _ = self.env.reset()
        spec_state = self.spec.reset()

        prod_state = np.concatenate([env_state, spec_state], 0)

        return prod_state
    
    def label_state(self, state):

        variables = {
            "a": state[2] < -0.087,
            "b": state[2] > 0.087
        }

        return variables
    
    def step(self, action):

        # Fourroom State
        env_state, env_reward, env_done, _, _  = self.env.step(action)

        # Spec State

        variables = self.label_state(env_state)
        variables["alive"] = env_done
        spec_state, spec_reward, spec_done = self.spec.step(**variables)

        # Prod State
        next_prod_state = np.concatenate([env_state, spec_state], 0)

        return next_prod_state, spec_reward, env_done, None