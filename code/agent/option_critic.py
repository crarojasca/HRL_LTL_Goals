import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.distributions import Categorical, Bernoulli

from torchvision.utils import save_image

import numpy as np
from math import exp

import random
from copy import deepcopy
from collections import deque

from omegaconf import OmegaConf

from .replay_memory import ReplayMemory


class Network(nn.Module):
    def __init__(self, in_features, num_actions, num_options, features_encoding="mlp", 
                 dims=[128, 64, 32], temperature=1.0, eps_start=1.0, eps_min=0.1, 
                 eps_decay=int(1e6), eps_test=0.05, device='cpu', testing=False):

        super(Network, self).__init__()
        
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
        self.eps = eps_start
        self.num_steps = 0

        self.features_encoding = features_encoding

        if not self.features_encoding:
            self.features = lambda x: x
            input_dim = in_features.shape[0]
        elif features_encoding=="mlp":
            self.features = nn.Sequential(
                nn.Linear(in_features.shape[0], dims[0]),
                nn.ReLU(),
                nn.Linear(dims[0], dims[1]),
                nn.ReLU()
            )
            input_dim = dims[1]   
        elif self.features_encoding=="conv":
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU()
            ).to(device)
            # self.features = lambda state: torch.cat(
            #     (env_features(state["env"]), state["spec"]))  
            input_dim = 512 #+ self.in_features["spec"].n

        self.Q            = nn.Linear(input_dim, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(input_dim, num_options)                 # Option-Termination
        self.options = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, dims[-1]),
                    nn.ReLU(),
                    nn.Linear(dims[-1], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_actions)
                ) for _ in range(num_options)
            ]
        )

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        # For the state sample case
        # if self.features_encoding=="conv":
        #     obs["env"] = torch.tensor(obs["env"], dtype=torch.float32, device=self.device)
        #     obs["env"] = obs["env"].permute(2, 0, 1)
        #     obs["spec"] = torch.tensor(obs["spec"], dtype=torch.float32, device=self.device)
        # else:

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
    
        if obs.ndim <= 3:
            obs = obs.unsqueeze(0)
        
        if self.features_encoding=="conv":
            obs = obs.permute(0, 3, 1, 2)
            obs = T.functional.crop(obs, 100, 40, 200, 280)
            # save_image(obs, 'img1.png')
            obs = T.Resize(size=(85, 85))(obs)
            
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
    
    def epsilon_exploration(self, probs):
        sample = random.random()
        
        if sample > self.eps:
            action_dist = Categorical(probs)
            action = probs.max(1)[1].view(1, 1)
            logp = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            return action.item(), logp, entropy, probs

        else:
            action = np.random.randint(0, self.num_actions)
            action = torch.tensor([[action]])
            logp = torch.log(torch.tensor([1/self.num_actions], device=self.device))
            entropy = torch.tensor([1], device=self.device)
            return action, logp, entropy, probs        

    def boltzmann_exploration(self, probs):
        action_dist = Categorical(probs)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy, probs

    def get_action(self, state, option):
        logits = self.options[option](state)
        probs = (logits / self.temperature).softmax(dim=-1)

        # if self.num_options==1:
        #     return self.epsilon_exploration(probs)
        # else:
        return self.boltzmann_exploration(probs)
        
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()
    
    def freeze(self):
        self.options_W.requires_grad = False
        self.options_b.requires_grad = False

    def update_epsilon(self):
        if not self.testing:
            self.eps = self.eps_min + (self.eps_start - self.eps_min)\
                  * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            self.eps = self.eps_test
        return self.eps
  
class OptionCritic:

    def __init__(self, observation_space, action_space, args) -> None:

        if args.seed:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        # Parameters
        self.args = args
        self.num_options = args.num_options
        self.max_episodes = args.max_episodes
        self.max_steps_ep = args.max_steps_ep
        self.update_frequency = args.update_frequency

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Main Network
        self.option_critic = Network(
            observation_space, action_space, args.num_options,
            features_encoding=args.network.features_encoding,
            dims=args.network.dimensions,
            temperature=args.temp,
            eps_start=args.epsilon.start,
            eps_min=args.epsilon.min,
            eps_decay=args.epsilon.decay,
            eps_test=args.epsilon.optimal,
            device=self.device
        )
        self.number_parameters = sum(p.numel() for p in self.option_critic.parameters())
        self.optim = torch.optim.RMSprop(self.option_critic.parameters(), lr=args.learning_rate)

        # Target Network
        self.option_critic_prime = deepcopy(self.option_critic)
        self.tau = args.target_network.tau
        self.freeze_interval = args.target_network.freeze_interval

        # Replay Memory
        self.batch_size = args.replay_memory.batch_size
        # self.buffer = ReplayBuffer(
        #     capacity=args.replay_memory.max_history, seed=args.seed)    
        
        self.buffer = ReplayMemory(
            capacity=args.replay_memory.max_history, 
            values=("obs", "option", "reward", "next_obs", "done"),
            seed=args.seed
        )
        
        # Counters
        self.steps = 0
        self.episodes = 0

    def save(self, conf, name):
        hyperparameters = OmegaConf.to_container(conf, resolve=True)
        torch.save(
            {
                'model_params': self.option_critic.state_dict(),
                'hyperparameters': hyperparameters
            }, os.path.join(conf.model.path, name)
        )

    def load(self, conf):
        if conf.model.load_model:
            checkpoint = torch.load(
                os.path.join(conf.model.path, conf.model.load_model))
            self.option_critic.load_state_dict(
                checkpoint['model_params'])

   
    def critic_loss_fn(self, model, model_prime, data_batch, args):
        obs, options, rewards, next_obs, dones = data_batch
        batch_idx = torch.arange(len(options)).long()
        options   = torch.LongTensor(options).to(model.device)
        rewards   = torch.FloatTensor(rewards).to(model.device)
        masks     = 1 - torch.FloatTensor(dones).to(model.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        states = model.get_state(obs).squeeze(0)
        Q      = model.get_Q(states)
        
        # the update target contains Q_next, but for stable learning we use prime network for this
        next_states_prime = model_prime.get_state(next_obs).squeeze(0)
        next_Q_prime      = model_prime.get_Q(next_states_prime) # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states            = model.get_state(next_obs).squeeze(0)
        next_termination_probs = model.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

        # Now we can calculate the update target gt
        gt = rewards + masks * args.gamma * \
            ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] \
             + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

        # to update Q we want to use the actual network, not the prime
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(Q[batch_idx, options], gt.detach().unsqueeze(1))
        loss = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()

        
        return loss

    def actor_loss_fn(self, obs, option, logp, entropy, reward, 
                      done, next_obs, model, model_prime, args):
        state = model.get_state(obs)
        next_state = model.get_state(next_obs)
        next_state_prime = model_prime.get_state(next_obs)

        option_term_prob = model.get_terminations(state)[:, option]
        next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

        Q = model.get_Q(state).detach().reshape([self.num_options])#.squeeze()
        next_Q_prime = model_prime.get_Q(next_state_prime).detach().reshape([self.num_options])#.squeeze()

        # Target update gt
        gt = reward + (1 - done) * args.gamma * \
            ((1 - next_option_term_prob) * next_Q_prime[option] \
              + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

        # The termination loss
        termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() \
                                                + args.termination_reg) * (1 - done)
        
        # actor-critic policy gradient with entropy regularization
        policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def run(self, env, logger=None):

        reward_list=[]

        while self.episodes < self.max_episodes:

            ep_reward = 0 ; option_lengths = {opt:[] for opt in range(self.num_options)}
            
            obs, info = env.reset()
            state = self.option_critic.get_state(obs)
            greedy_option  = self.option_critic.greedy_option(state)
            current_option = 0

            done = False ; truncated=False
            ep_steps = 0 ; option_termination = True ; curr_op_len = 0
                      
            while not(done or truncated) and ep_steps < self.max_steps_ep:
                
                epsilon = self.option_critic.update_epsilon()

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(
                        self.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0
                action, logp, entropy, probs = self.option_critic.get_action(state, current_option)

                next_obs, reward, done, truncated, _ = env.step(action)

                self.buffer.push(obs, current_option, reward, next_obs, done)
                ep_reward += reward

                actor_loss, critic_loss = None, None
                if len(self.buffer) > self.batch_size:
                    actor_loss = self.actor_loss_fn(obs, current_option, logp, entropy, \
                        reward, done, next_obs, self.option_critic, self.option_critic_prime, self.args)
                    loss = actor_loss

                    if self.steps % self.update_frequency == 0:
                        data_batch = self.buffer.sample(self.batch_size)
                        critic_loss = self.critic_loss_fn(
                            self.option_critic, self.option_critic_prime, data_batch, self.args)
                        loss += critic_loss

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    if self.steps % self.freeze_interval == 0:
                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        net_state_dict = self.option_critic.state_dict()
                        target_net_state_dict = self.option_critic_prime.state_dict()
                        for key in net_state_dict:
                            target_net_state_dict[key] = net_state_dict[key]*self.tau + \
                                target_net_state_dict[key]*(1-self.tau)
                        self.option_critic_prime.load_state_dict(target_net_state_dict)

                state = self.option_critic.get_state(next_obs)
                option_termination, greedy_option = self.option_critic.predict_option_termination(
                    state, current_option)

                # update global steps etc
                self.steps += 1
                ep_steps += 1
                curr_op_len += 1
                obs = next_obs


                if logger:      
                    logger.log_data(
                        self.steps, reward, actor_loss, 
                        critic_loss, entropy.item(), epsilon)

            reward_list += [ep_reward]
            mean_reward = np.mean(reward_list[-100:])
            self.episodes += 1

            if logger:
                logger.log_episode(
                    self.steps, ep_steps, self.episodes, ep_reward, mean_reward, epsilon, option_lengths)