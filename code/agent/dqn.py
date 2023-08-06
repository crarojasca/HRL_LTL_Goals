import numpy as np
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', "done"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Network(nn.Module):

    def __init__(self, observation_space, action_space, features_encoding=None):
        super(Network, self).__init__()

        self.features_encoding = features_encoding
        if self.features_encoding == "mlp":
            self.Q_input_dim = 64
            self.features = nn.Sequential(
                nn.Linear(observation_space, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )
        else:
            self.Q_input_dim = observation_space
            self.features = lambda x:x


        self.Q1 = nn.Linear(self.Q_input_dim, 64)
        self.Q2 = nn.Linear(64, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.Q1(x))
        return self.Q2(x)
    
    def get_state(self, x):
        if x.ndim < 4:
            x = x.unsqueeze(0)
        return self.features(x)
    
    
    

class DQN:

    def __init__(self, observation_space, action_space, logger, args):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.SystemRandom(args.seed)

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = Network(observation_space, action_space, args.features_encoding).to(self.device)
        self.target_net = Network(observation_space, action_space, args.features_encoding).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if not args.features_encoding:
            self.features = lambda x: x
            input_dim = observation_space
        elif args.features_encoding=="mlp":
            self.features = self.policy_net.get_state
            input_dim = 64  

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=args.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(args.max_history)

        self.action_space = action_space


        self.logger = logger
        self.max_episodes = args.max_episodes
        self.max_steps_ep = args.max_steps_ep

        self.epsilon_start = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay

        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.gamma
        self.TAU = args.tau
        self.freeze_interval = args.freeze_interval

        self.steps_done = 0
        self.epsilon = None

        self.episode_durations = []

    def random_action(self):
        action = np.random.randint(0, self.action_space)
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        return action
    
    def greedy_action(self, state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)

    def choose_action(self, state):
        sample = random.random()
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if sample > self.epsilon:
            return self.greedy_action(state)
        else:
            return self.random_action()
        
    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None
        
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        done = torch.tensor(batch.done, dtype=torch.int16, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (1-done)*self.GAMMA*next_state_values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def run(self, env):
        
        self.steps_done = 0
        rewards = []

        for episode in range(self.max_episodes):

            # Initialize the environment and get it's state
            observation, info = env.reset()
            state = self.features(torch.tensor(observation, dtype=torch.float32, device=self.device)).unsqueeze(0)
            ep_reward = 0
            ep_steps = 0

            done=False; truncated=False
            while not(done or truncated):
                
                ep_steps += 1
                self.steps_done += 1

                action = self.choose_action(state)

                observation, reward, done, truncated, _ = env.step(action.item())

                reward = torch.tensor([reward], device=self.device)
                ep_reward += reward
                
                next_state = self.features(torch.tensor(observation, dtype=torch.float32, device=self.device)).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                if self.steps_done % self.freeze_interval == 0:
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                self.logger.log_data(
                    self.steps_done, reward, 0, loss, 0, self.epsilon, action, 0)
            
            rewards += [ep_reward.cpu().item()]
            mean_reward = np.mean(rewards[-100:])
            self.logger.log_episode(self.steps_done, ep_steps, episode, ep_reward.item(), mean_reward, self.epsilon, None)


if __name__ == "__main__":
    from main import dqn_args, log_args
    from logger import Logger

    env = gym.make("CartPole-v1")
    logger = Logger(
        logdir=log_args.logdir, 
        run_name=f"{dqn_args.name}-cart-pole")

    agent = DQN(
        observation_space=env.observation_space.shape[0], 
        action_space=env.action_space.n,
        logger=logger,
        args=dqn_args
    )
    agent.run(env)