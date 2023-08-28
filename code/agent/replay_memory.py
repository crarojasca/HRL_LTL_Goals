import random
import torch as T
import numpy as np
from collections import namedtuple, deque

class ReplayMemory(object):

    def __init__(self, capacity, 
                 values=('obs', 'action', 'next_obs', 'reward', "done"), seed=42):
        random.seed(seed)
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', values)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        transitions =  random.sample(self.memory, batch_size)
        transitions = [np.array(value) for value in zip(*transitions)]
        batch = self.transition(*transitions)
        return batch
    
    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)