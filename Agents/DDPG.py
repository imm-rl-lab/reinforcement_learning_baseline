import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
from Agents.Utilities.LinearTransformations import transform_interval


class DDPG:
    def __init__(self, action_min, action_max, q_model, pi_model, noise,
                 q_model_lr=1e-3, pi_model_lr=1e-4, gamma=0.99, batch_size=64, tau=1e-3,
                 memory_len=100000):

        self.action_min = action_min
        self.action_max = action_max
        self.q_model = q_model
        self.pi_model = pi_model
        self.noise = noise
        
        self.q_model_lr = q_model_lr
        self.pi_model_lr = pi_model_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        self.memory = deque(maxlen=memory_len)
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.q_target_model = deepcopy(self.q_model)
        self.pi_target_model = deepcopy(self.pi_model)
        return None

    def get_action(self, state):
        action = self.pi_model(state).detach().numpy() + self.noise.get()
        action = self.transform_interval(action)
        return np.clip(action, self.action_min, self.action_max)
    
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
            pred_next_actions = self.transform_interval(self.pi_target_model(next_states))
            next_q_values = self.get_q_values(self.q_target_model, next_states, pred_next_actions)
            targets = rewards + (1 - dones) * self.gamma * next_q_values
            
            #train q_model
            q_values = self.get_q_values(self.q_model, states, actions)
            q_loss = torch.mean((q_values - targets.detach()) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.q_optimizer, q_loss)

            #train u_pi_model
            pred_actions = self.transform_interval(self.pi_model(states))
            q_values = self.get_q_values(self.q_model, states, pred_actions)
            pi_loss = - torch.mean(q_values)
            self.update_target_model(self.pi_target_model, self.pi_model, self.pi_optimizer, pi_loss)

        return None
    
    def get_q_values(self, q_model, states, actions):
        states_and_actions = torch.cat((states, actions), dim=1)
        return q_model(states_and_actions)
    
    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None

    def reset(self):
        self.noise.reset()
        self.noise.reduce()
        return None
