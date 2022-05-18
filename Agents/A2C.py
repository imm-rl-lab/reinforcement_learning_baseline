import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agents.Utilities.Noises import DiscreteUniformNoise


class A2C():
    def __init__(self, state_dim, action_min, action_max, pi_model, v_model, 
                 gamma=0.99, pi_model_lr=1e-3, v_model_lr=1e-3, entropy_threshold=1, action_space_type='discrete'):
        self.state_dim = state_dim
        self.action_min = action_min
        self.action_max = action_max
        self.pi_model = pi_model
        self.v_model = v_model
        self.noise = DiscreteUniformNoise(2)
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold
        self.action_space_type = action_space_type

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_model_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), v_model_lr)
        
        return None
    
    def reset(self):
        pass
    
    def get_dist(self, states):
        if self.action_space_type == 'discrete':
            logits = self.pi_model(states)
            probs = F.softmax(logits, -1)
            return torch.distributions.Categorical(probs)
        if self.action_space_type == 'continuous':
            mean = self.pi_model(states)
            sigma = torch.full(mean.size(), 0.1)
            return torch.distributions.Normal(mean, 0.1)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        dist = self.get_dist(state)
        if self.action_space_type == 'discrete':
            action = int(dist.sample())
        else:
            action = dist.sample().detach().data.numpy()
            action = np.clip(action, self.action_min, self.action_max)
        return action

    def get_returns(self, rewards, last_done, last_value):
        returns = rewards.copy()
        returns[-1] += self.gamma * (1 - last_done) * last_value
        for i in range(len(rewards) - 2, -1, -1):
            returns[i] += self.gamma * returns[i + 1]
        return returns

    def fit(self, sessions):
        
        states, actions, rewards, returns = [], [], [], []
        for session in sessions:
            states.extend(session['states'][:-1])
            actions.extend(session['actions'])
            rewards.extend(session['rewards'])
            
            last_done = session['dones'][-1]
            last_state = session['states'][-1]
            last_value = self.v_model(torch.FloatTensor(last_state))
            returns.extend(self.get_returns(session['rewards'], last_done, last_value))
        
        if self.action_space_type == 'discrete':
            actions = torch.LongTensor(actions)
        if self.action_space_type == 'continuous':
            actions = torch.FloatTensor(actions)
        
        states = torch.FloatTensor(states)
        returns = torch.FloatTensor(returns)
        
        advantages = returns.detach() - self.v_model(states)
        
        dist = self.get_dist(states)
        entropy = dist.entropy().mean()
        
        pi_loss = - (torch.mean(dist.log_prob(actions) * advantages.detach()) + self.entropy_threshold * entropy)
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        
        v_loss = torch.mean((returns - self.v_model(states)) ** 2)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        return None
