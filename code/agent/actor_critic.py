import numpy as np
from .base import ReplayMemory

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, observation_space, n_actions, features_encoding=None):
        super(NN, self).__init__()

        self.features_encoding = features_encoding

        self.base_layers = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)

    def forward(self, state):
        x = self.base_layers(state)
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)
    

class ActorCritic():
    def __init__(self, observation_space, action_space, args=None):

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        # Parameters
        self.gamma = args.gamma
        self.tau = args.target_network.tau
        self.batch_size = args.replay_memory.batch_size
        self.max_episodes = args.max_episodes
        self.max_steps_ep = args.max_steps_ep
        self.log_probs = []

        # Replay Memory
        self.memory = ReplayMemory(
            args.replay_memory.max_history, 
            values=('state', 'logp', 'next_state', 'reward', "done"), seed=42)
        
        # Networks
        self.network = NN(observation_space, action_space).to(self.device)
        self.target_network = NN(observation_space, action_space).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.number_parameters = sum(p.numel() for p in self.network.parameters())        


    def choose_action(self, state):
        probabilities, _ = self.network.forward(state)
        probabilities = F.softmax(probabilities, dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        logp = action_probs.log_prob(action).view(1, 1)
        entropy = action_probs.entropy()

        return action.item(), logp, entropy

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None, None

        state, logp, reward, next_state, done = \
                                self.memory.sample(self.batch_size)

        # Get Critic Values
        with T.no_grad():
            _, critic_value_ = self.target_network.forward(next_state)
        _, critic_value = self.network.forward(state)

        # Compute delta
        delta = reward + (1-done)*self.gamma*critic_value_

        actor_loss = -T.mean(logp*(delta-critic_value))
        critic_loss = F.mse_loss(delta, critic_value)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss, critic_loss

    def run(self, env, logger=None):

        steps = 0
        episodes = 0
        reward_list=[]


        while episodes < self.max_episodes:

            ep_reward = 0

            obs, info = env.reset()
            state = T.tensor(obs, dtype=T.float32, device=self.device).view(1, -1)

            done = False ; ep_steps = 0
                      
            while not done and ep_steps < self.max_steps_ep:
                        
                action, logp, entropy = self.choose_action(state)

                # Take action in the env
                next_obs, reward, done, truncated, info = env.step(action)
                # Set values
                next_state = T.tensor(next_obs, dtype=T.float32, device=self.device).view(1, -1)
                reward = T.tensor(reward, device=self.device).view(1, 1)
                done = T.tensor(done, dtype=T.int8, device=self.device).view(1, 1)
                # Store transition in buffer
                self.memory.push(
                    state, logp, reward, next_state, done)
                # Update NN parameters
                actor_loss, critic_loss = self.optimize()

                target_net_state_dict = self.target_network.state_dict()
                net_state_dict = self.network.state_dict()
                for key in net_state_dict:
                    target_net_state_dict[key] = net_state_dict[key]*self.tau \
                            + target_net_state_dict[key]*(1-self.tau)
                self.target_network.load_state_dict(target_net_state_dict)

                # Update counters
                ep_reward += reward    
                steps += 1
                ep_steps += 1
                obs = next_obs

                if logger:
                    logger.log_data(
                        steps, reward, actor_loss, critic_loss, entropy, 0)
                            

            reward_list += [ep_reward.cpu().item()]
            mean_reward = np.mean(reward_list[-100:])
            episodes += 1

            if logger:
                logger.log_episode(steps, ep_steps, episodes, ep_reward.item(), mean_reward, 0)