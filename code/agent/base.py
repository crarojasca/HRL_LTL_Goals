import random
import torch as T
import numpy as np
from collections import namedtuple, deque

class ReplayMemory(object):

    def __init__(self, capacity, device,
                 values=('obs', 'action', 'next_obs', 'reward', "done"), seed=42):
        random.seed(10)
        self.device = device
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', values)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        transitions =  random.sample(self.memory, batch_size)
        transitions = [
            T.tensor(np.array(value), dtype=T.float32, device=self.device).view(batch_size, -1) \
                for value in zip(*transitions)]
        batch = self.transition(*transitions)
        return batch

    def __len__(self):
        return len(self.memory)