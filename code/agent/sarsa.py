import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


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


class Q(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden=64):
        super(Q, self).__init__()
        self.fc1= nn.Linear(observation_space, hidden)
        self.fc2= nn.Linear(hidden, action_space)

    def forward(self, state):

        x = self.fc1(state)
        x = F.relu(x)
        q = self.fc2(x)
        return q

class Sarsa():
    def __init__(self, observation_space, action_space, logger, args):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.SystemRandom(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q=Q(observation_space, action_space).to(self.device)
        self.Q_target=Q(observation_space, action_space).to(self.device)

        self.action_space = action_space

        self.steps=0
        self.optimizer=optim.Adam(self.Q.parameters(), lr=args.learning_rate)
        self.gamma = args.gamma
        self.epsilon_test = args.epsilon_test
        self.eps_min = args.epsilon_min
        self.eps_start = args.epsilon_start
        self.eps_decay = args.epsilon_decay
        self.eps_test  = args.epsilon_test
        self.eps = args.epsilon_start
        self.freeze_interval = args.freeze_interval
        self.batch_size = args.batch_size
        self.buffer = ReplayMemory(args.max_history)

        self.testing = args.testing

        self.logger = logger
        self.max_episodes = args.max_episodes
        self.max_steps_ep = args.max_steps_ep
        

    def optimize(self, state, action, next_state, reward, done):
        q = self.Q(state)[action]
        q_targets = self.Q_target(next_state)

        action_target = self.greedy_action(next_state)
        
        q_target = q_targets[action_target]

        y = reward + (1-done) * self.gamma*q_target

        loss=F.mse_loss(y, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps%self.freeze_interval==0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return loss

    def greedy_action(self, state):
        q = self.Q(state)
        return q.argmax(dim=-1).view(-1, 1)

    def random_action(self):
        action = np.random.randint(0, self.action_space)
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        return action

    def choose_action(self, state):
        if np.random.rand()> self.eps:
            return self.random_action()
        else:
            return self.greedy_action(state)
        
    def get_state(self, obs):
        return torch.tensor(obs, dtype=torch.float32).view(1, -1).to(self.device)
    
    def update_epsilon(self):
        if not self.testing:
            self.eps = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-self.steps / self.eps_decay)
        else:
            self.eps = self.eps_test
        
    def run(self, env):
        self.steps = 0
        episodes = 0
        reward_list=[]

        while episodes < self.max_episodes:

            ep_reward = 0

            obs, _ = env.reset()

            state = self.get_state(obs)

            done = False ; ep_steps = 0 ; critic_loss = 0

            action = self.choose_action(state)
                      
            while not done and ep_steps < self.max_steps_ep:
                        
                next_obs, reward, done, _, _ = env.step(action)
                next_state = self.get_state(next_obs)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(1, 1)
                done = torch.tensor(done, dtype=torch.int16, device=self.device).view(1, 1)

                next_action = self.choose_action(next_state)

                self.buffer.push(state, action, next_state, reward, done)
                
                if len(self.buffer) > self.batch_size:
                    
                    transitions = self.buffer.sample(self.batch_size)
                    # print(transitions)

                    batch = Transition(*zip(*transitions))

                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    next_state_batch = torch.cat(batch.next_state)
                    reward_batch = torch.cat(batch.reward)
                    done_batch = torch.cat(batch.done)

                    critic_loss = self.optimize(
                        state_batch, action_batch, next_state_batch, reward_batch, done_batch)
                    
                self.update_epsilon()

                ep_reward += reward.item()    
                self.steps += 1
                ep_steps += 1
                obs = next_obs

                            
                self.logger.log_data(
                    self.steps, reward.item(), 0, critic_loss, 0, 0, action, 0)
                
                action = next_action
                

            reward_list += [ep_reward]
            mean_reward = np.mean(reward_list[-100:])
            episodes += 1
            self.logger.log_episode(self.steps, ep_steps, episodes, ep_reward, mean_reward, self.eps, None)