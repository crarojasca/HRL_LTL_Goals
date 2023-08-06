import math
import numpy as np
from scipy.special import softmax

import torch
from torch.distributions import Categorical


# FrozenLake-v0 gym environment
# env = gym.make('FrozenLake-v0')

# Parameters
# epsilon = 0.9
# total_episodes = 10000
# max_steps = 100
# alpha = 0.05
# gamma = 0.95

class Q:

    def __init__(self, n_observations, n_actions):
        self.n_actions = n_actions
        self.values = {}

    def __call__(self, state):

        if not state in self.values:
            self.values[state] = torch.zeros(self.n_actions)
        
        return self.values[state]


  
class Sarsa:

    def __init__(self, observation_space, action_space, logger, args):

        np.random.seed(args.seed)

        #Initializing the Q-value
        self.Q=Q(observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        self.iter_num=0
        self.gamma = args.gamma
        self.epsilon_min = args.epsilon_min

        self.epsilon_start = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay

        self.gamma = args.gamma
        self.lr = args.learning_rate

        self.logger = logger
        self.temperature = args.temperature
        self.max_episodes = args.max_episodes
        self.max_steps_ep = args.max_steps_ep

    # Function to choose the next action with episolon greedy
    def choose_action(self, state):

        # self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
        #     math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        # if np.random.uniform(0, 1) > self.epsilon:
        #     action = np.random.randint(0, self.action_space)
        # else:
        logits = self.Q(state)
        probs = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(probs)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy, logits
    
    def get_state(self, obs):
        return (torch.tensor(obs) * torch.arange(1, self.observation_space+1)).sum()
    
    def optimize(self, state, action, reward, next_state, next_action):
        #               S     A       R       S'           A'
        #Learning the Q-value
        self.Q.values[state][action] = self.Q(state)[action] + \
            self.lr * (reward + self.gamma * self.Q(next_state)[next_action] - self.Q(state)[action])
    
    def run(self, env):

        self.steps_done = 0
        rewards = []

        # Starting the SARSA learning
        for episode in range(self.max_episodes):
            observation, info = env.reset()
            state = self.get_state(observation)
            action, logp, entropy, logits  = self.choose_action(state)
        
            eps_reward = 0
            eps_steps = 0
            done=False; truncated=False
            while not(done or truncated) and eps_steps<self.max_steps_ep:

                eps_steps += 1
                self.steps_done += 1
                
                # Getting the next state
                observation, reward, done, truncated, info = env.step(action)
                next_state = self.get_state(observation)
        
                #Choosing the next action
                next_action, logp, entropy, logits = self.choose_action(next_state)
                
                self.optimize(state, action, reward, next_action, next_action)

                state = next_state
                action = next_action
                eps_reward += reward
                
                self.logger.log_data(
                    self.steps_done, reward, 0, 0, 0, 0, 0, 0)
                
            rewards += [eps_reward]
            mean_reward = np.mean(rewards[-100:])
            self.logger.log_episode(episode, eps_reward, mean_reward, {}, eps_steps, 0)