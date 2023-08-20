import numpy as np
from .replay_memory import ReplayMemory

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, observation_space, n_actions, device, dims=[128, 64, 32], 
                 features_encoding=None):
        super(NN, self).__init__()

        self.device = device

        self.observation_space = observation_space
        self.features_encoding = features_encoding

        # self.base_layers = nn.Sequential(
        #     nn.Linear(observation_space, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU()
        # )
        # self.policy = nn.Linear(64, n_actions)
        # self.value = nn.Linear(64, 1)

        if not self.features_encoding:
            self.features = lambda x: x
            input_dim = observation_space.shape[0]
        elif features_encoding=="mlp":
            self.features = nn.Sequential(
                nn.Linear(observation_space.shape[0], dims[0]),
                nn.ReLU(),
                nn.Linear(dims[0], dims[1]),
                nn.ReLU()
            )
            input_dim = dims[1]  

        print(input_dim)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], n_actions),
        )

        self.Q = nn.Sequential(
            nn.Linear(input_dim, dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], 1),
        )

        self.to(self.device)

    def forward(self, obs):
        state = self.get_state(obs)
        action, logp, entropy = self.get_action(state)
        Q = self.get_Q(state)
        return action, logp, entropy, Q
    
    def get_state(self, obs):
        obs = T.tensor(obs, dtype=T.float32, device=self.device) \
            .view(-1, self.observation_space.shape[0])
        state = self.features(obs)
        return state
    
    def get_Q(self, state):
        Q = self.Q(state)
        return Q
    
    def get_action(self, state):
        # probabilities, _ = self.network.forward(state)
        logits = self.policy(state)
        probabilities = F.softmax(logits, dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        logp = action_probs.log_prob(action)
        entropy = action_probs.entropy()

        return action, logp, entropy
    
    

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
            values=('obs', 'logp', 'next_obs', 'reward', "done"), seed=42)
        
        # Networks
        self.network = NN(observation_space, action_space, self.device, 
                          features_encoding=args.network.features_encoding,
                          dims=args.network.dimensions)
        self.target_network = NN(observation_space, action_space, self.device,
                                 features_encoding=args.network.features_encoding,
                                dims=args.network.dimensions)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.number_parameters = sum(p.numel() for p in self.network.parameters())        


    # def choose_action(self, state):
    #     probabilities, _ = self.network.forward(state)
    #     probabilities = F.softmax(probabilities, dim=-1)
    #     action_probs = T.distributions.Categorical(probabilities)
    #     action = action_probs.sample()
    #     logp = action_probs.log_prob(action).cpu().detach().numpy()
    #     entropy = action_probs.entropy()

    #     return action.item(), logp, entropy

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None, None

        obs, logp, reward, next_obs, done = \
                                self.memory.sample(self.batch_size)
        
        # Cast the variables
        obs = T.tensor(obs, dtype=T.float32, device=self.device)
        logp = T.tensor(logp, dtype=T.float32, device=self.device)
        reward = T.tensor(reward, dtype=T.float32, device=self.device)
        next_obs = T.tensor(next_obs, dtype=T.float32, device=self.device)
        done = T.tensor(done, dtype=T.int8, device=self.device)

        # Get Critic Values
        with T.no_grad():
            _, _, _, Q_ = self.target_network.forward(next_obs)
        _, _, _, Q  = self.network.forward(obs)
        
        # Compute delta
        delta = reward + (1-done.type(T.int8))*self.gamma*Q_

        actor_loss = -T.mean(logp*(delta - Q))
        critic_loss = F.mse_loss(delta, Q)

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

            done = False ; ep_steps = 0
                      
            while not done and ep_steps < self.max_steps_ep:
                        
                # action, logp, entropy = self.network.get_action(state)
                action, logp, entropy, Q = self.network(obs)

                # Take action in the env
                next_obs, reward, done, truncated, info = env.step(action.item())
        
                # Store transition in buffer
                self.memory.push(
                    obs, logp.cpu().detach().numpy(), reward, next_obs, done)
                
                # Optimize the main network
                actor_loss, critic_loss = self.optimize()

                # Update NN parameters of the target network
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

                # Log step
                if logger:
                    logger.log_data(
                        steps, reward, actor_loss, critic_loss, entropy, 0)
                            

            reward_list += [ep_reward]
            mean_reward = np.mean(reward_list[-100:])
            episodes += 1

            if logger:
                logger.log_episode(steps, ep_steps, episodes, ep_reward, mean_reward, 0)