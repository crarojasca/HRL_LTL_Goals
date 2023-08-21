import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

import random
from copy import deepcopy
from .logger import Logger
from collections import deque

from .utils import to_tensor

import pygame

class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(obs), option, reward, np.stack(next_obs), done

    def __len__(self):
        return len(self.buffer)


class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )

        self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()
    
    def freeze(self):
        self.options_W.requires_grad = False
        self.options_b.requires_grad = False

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticFeatures, self).__init__()
        
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q            = nn.Linear(64, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        probs = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(probs)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy, logits
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()
    
    def freeze(self):
        self.options_W.requires_grad = False
        self.options_b.requires_grad = False

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss_fn(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q      = model.get_Q(states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime) # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss_fn(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss


    
class Agent:

    def __init__(self, args, max_episodes) -> None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.num_options = args.num_options

        option_critic = OptionCriticFeatures
        device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

        self.option_critic = option_critic(
            in_features=1,
            num_actions=3,
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device
        )
        
        # Create a prime network for more stable Q values
        self.option_critic_prime = deepcopy(self.option_critic)

        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=args.learning_rate)

        self.buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
        self.logger = Logger(
            logdir=args.logdir, 
            run_name=f"{args.name}-{args.env}-{self.num_options}_opts")
        
        self.max_steps_ep = args.max_steps_ep

        self.max_steps_total = args.max_steps_total

        self.batch_size = args.batch_size

        self.batch_size = args.batch_size

        self.freeze_interval = args.freeze_interval

        self.args = args

        self.update_frequency = args.update_frequency

        self.max_episodes = max_episodes

    def init(self, nstates, nactions):
        pass

    def set_action_names(self, action_names):
        pass

    def freeze_options(self):
        pass

    def run(self, env):

        pygame.key.set_repeat(10)
        def key_pressed(key):
            for event in pygame.event.get(): 
                if event.type == pygame.KEYDOWN and event.key == key: 
                    return True
            return False
        
        play = True   

        steps = 0
        episodes = 0
        # if args.switch_goal: print(f"Current goal {env.cumreward}")
        # while steps < self.max_steps_total:
        while episodes < self.max_episodes:

            rewards = 0 ; option_lengths = {opt:[] for opt in range(self.num_options)}

            
            env.reset()
            env.draw()
            obs   = [env.getstate()]
            state = self.option_critic.get_state(to_tensor(obs))
            greedy_option  = self.option_critic.greedy_option(state)
            current_option = 0

            # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
            # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
            # should be finedtuned (this is what we would hope).
            # if args.switch_goal and logger.n_eps == 1000:
            #     torch.save({'model_params': option_critic.state_dict(),
            #                 'goal_state': env.cumreward},
            #                 f'models/{args.name}_{args.num_options}_{args.seed}_1k')
            #     env.switch_goal()
            #     option_critic_prime.freeze()
            #     option_critic.freeze()
            #     print(f"New goal {env.goal}")


            # if self.args.switch_goal and self.logger.n_eps > 2000:
            #     torch.save({'model_params': self.option_critic.state_dict(),
            #                 'goal_state': env.cumreward},
            #                 f'models/{self.args.name}_{self.num_options}_{self.args.seed}_{steps}')
            #     # break

            done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0

                      
            while not done and ep_steps < self.max_steps_ep:
                
                if key_pressed(pygame.K_n): play = False
                while not play:
                    if key_pressed(pygame.K_n): play = True
                    if key_pressed(pygame.K_m): break


                epsilon = self.option_critic.epsilon

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(
                        self.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0
        
                action, logp, entropy, probs = self.option_critic.get_action(state, current_option)

                env.update(action)
                next_obs = [env.getstate()]
                reward = env.getreward()
                done = env.finished
                env.draw()
                # next_obs, reward, done, _ = env.step(action)
                self.buffer.push(obs, current_option, reward, next_obs, done)
                rewards += reward

                actor_loss, critic_loss = None, None
                if len(self.buffer) > self.batch_size:
                    actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                        reward, done, next_obs, self.option_critic, self.option_critic_prime, self.args)
                    loss = actor_loss

                    if steps % self.update_frequency == 0:
                        data_batch = self.buffer.sample(self.batch_size)
                        critic_loss = critic_loss_fn(
                            self.option_critic, self.option_critic_prime, data_batch, self.args)
                        loss += critic_loss

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    if steps % self.freeze_interval == 0:
                        self.option_critic_prime.load_state_dict(self.option_critic.state_dict())

                state = self.option_critic.get_state(to_tensor(next_obs))
                option_termination, greedy_option = self.option_critic.predict_option_termination(
                    state, current_option)

                # update global steps etc
                steps += 1
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs

                            
                self.logger.log_data(
                    steps, rewards, actor_loss, critic_loss, entropy.item(), epsilon, action, probs)

            episodes += 1
            self.logger.log_episode(episodes, rewards, option_lengths, ep_steps, epsilon)