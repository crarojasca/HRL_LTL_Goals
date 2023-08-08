import torch
import random
from collections import namedtuple, deque


class ReplayMemory(object):

    def __init__(self, capacity,
                 values=('state', 'action', 'next_state', 'reward', "done"), seed=42):
        random.seed(10)
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', values)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        transitions =  random.sample(self.memory, batch_size)
        transitions = [torch.cat(value).detach() for value in zip(*transitions)]
        batch = self.transition(*transitions)
        return batch

    def __len__(self):
        return len(self.memory)