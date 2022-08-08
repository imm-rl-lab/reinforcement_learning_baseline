import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from copy import deepcopy
from collections import deque
from Agents.Utilities.LinearTransformations import transform_interval


class SAC:
    def __init__(self, action_min, action_max, q1_model, q2_model, pi_model, noise,
                 q_model_lr=1e-3, pi_model_lr=1e-4, gamma=0.99, batch_size=64, tau=1e-3,
                 entropy_coef='auto', memory_len=1000000):

        self.action_min = action_min
        self.action_max = action_max
        self.q1_model = q1_model
        self.q2_model = q2_model
        self.pi_model = pi_model
        self.noise = noise
        
        self.q_model_lr = q_model_lr
        self.pi_model_lr = pi_model_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.entropy_coef = entropy_coef
        
        self.memory = deque(maxlen=memory_len)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), lr=q_model_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), lr=q_model_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)
        self.pi_target_model = deepcopy(self.pi_model)
        
        if self.entropy_coef=='auto':
            self.target_entropy = - torch.FloatTensor([self.action_min.shape[0]])
            print(self.target_entropy)
            self.log_entropy_coef = torch.zeros(1, requires_grad=True)
            self.entropy_coef_optimizer = torch.optim.Adam([self.log_entropy_coef], lr=1e-3)
            
        return None

    def get_action(self, state):
        action, _, action_mean = self.get_actions_and_log_probs(self.pi_model, [state])
        if self.noise.threshold == 0:
            return np.clip(action_mean.squeeze(0).data.numpy(), self.action_min, self.action_max)
        else:
            return np.clip(action.squeeze(0).data.numpy(), self.action_min, self.action_max)
    
    def transform_interval(self, action):
        return transform_interval(action, self.action_min, self.action_max)
    
    def fit(self, state, action, reward, done, next_state):
        #add to memory
        self.memory.append([state, action, reward, done, next_state])
        
        if len(self.memory) >= self.batch_size:
            #get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            #get targets
            pred_next_actions, next_log_probs, _ = self.get_actions_and_log_probs(self.pi_target_model, next_states)
            next_q1_values = self.get_q_values(self.q1_target_model, next_states, pred_next_actions)
            next_q2_values = self.get_q_values(self.q2_target_model, next_states, pred_next_actions)
            
            if self.entropy_coef=='auto':
                entropy_coef = torch.exp(self.log_entropy_coef).detach()
            else:
                entropy_coef = self.entropy_coef

            targets = rewards + (1 - dones) * self.gamma * (torch.minimum(next_q1_values, next_q2_values) - entropy_coef * next_log_probs)
            
            #train q1_model
            q1_values = self.get_q_values(self.q1_model, states, actions)
            q1_loss = torch.mean((q1_values - targets.detach()) ** 2)
            self.update_target_model(self.q1_target_model, self.q1_model, self.q1_optimizer, q1_loss, self.tau)
            
            #train q2_model
            q2_values = self.get_q_values(self.q2_model, states, actions)
            q2_loss = torch.mean((q2_values - targets.detach()) ** 2)
            self.update_target_model(self.q2_target_model, self.q2_model, self.q2_optimizer, q2_loss, self.tau)

            #train u_pi_model
            pred_actions, log_probs, _ = self.get_actions_and_log_probs(self.pi_model, states)
            q1_values = self.get_q_values(self.q1_model, states, pred_actions)
            q2_values = self.get_q_values(self.q2_model, states, pred_actions)
            pi_loss = - torch.mean(torch.min(q1_values, q2_values) - entropy_coef * log_probs)
            self.update_target_model(self.pi_target_model, self.pi_model, self.pi_optimizer, pi_loss, self.tau)
            
            #train entropy_coef
            if self.entropy_coef=='auto':
                entropy_loss = - (self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
                self.entropy_coef_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_coef_optimizer.step()

        return None
    
    def get_q_values(self, q_model, states, actions):
        states_and_actions = torch.cat((states, actions), dim=1)
        return q_model(states_and_actions)
    
    def update_target_model(self, target_model, model, optimizer, loss, tau):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        return None

    def reset(self):
        self.noise.reset()
        self.noise.reduce()
        return None

    def get_actions_and_log_probs(self, pi_model, states):
        means, log_stds = pi_model(states)
        log_stds = torch.clamp(log_stds, -20, 2)
        dist = Normal(means, torch.exp(log_stds))
        sample = dist.rsample()
        actions = self.transform_interval(torch.tanh(sample))
        log_probs = dist.log_prob(sample)
        action_scale = torch.FloatTensor(self.action_max - self.action_min) / 2
        log_probs -= torch.log(action_scale * (1 - torch.tanh(sample).pow(2)) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        action_means = self.transform_interval(torch.tanh(means))
        return actions, log_probs, action_means
        