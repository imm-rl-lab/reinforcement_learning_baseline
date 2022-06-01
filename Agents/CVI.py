import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from Agents.Utilities.LinearTransformations import transform_interval


class VModelWithGradient(nn.Module):
    def __init__(self, action_min, action_max, v_model, r, g):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        self.v_model = v_model
        self.r = r
        self.g = g
        return None

    def forward(self, state):
        return self.v_model(state)

    def get_gradient_actions(self, states):
        states = torch.FloatTensor(states)
        states.requires_grad = True
        values = self.v_model(states)
        values.backward(gradient=torch.ones_like(values))
        gradient_values = states.grad[:, 1:].detach().unsqueeze(2)
        g_values = torch.FloatTensor([self.g(state) for state in states])
        actions = 0.5 * (1 / self.r) * torch.matmul(g_values, gradient_values)[:, :, 0]
        return np.clip(actions.squeeze(0).detach().numpy(), self.action_min, self.action_max)


class CVI:
    def __init__(self, action_min, action_max, v_model, noise, virtual_step_for_batch, 
                 batch_size=128, gamma=1, tau=1e-3, v_model_lr=1e-3, predicted_step_n=4, memory_size=100000):
        self.action_max = action_max
        self.action_min = action_min
        self.v_model = v_model
        self.noise = noise
        self.virtual_step_for_batch = virtual_step_for_batch
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.predicted_step_n = predicted_step_n
        
        self.memory = deque(maxlen=memory_size)
        self.optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_model_lr)
        self.v_target_model = deepcopy(self.v_model)
        return None

    def get_action(self, state):
        gradient_action = self.v_target_model.get_gradient_actions([state])
        action = gradient_action + self.noise.get()
        return action

    def fit(self, state, action, reward, done, next_state):
        #save state in memory
        if not done:
            self.memory.append(state)

        if len(self.memory) >= self.batch_size:
            #choose states from memory
            states = np.array(random.sample(self.memory, self.batch_size))
            
            #calculate targets
            targets = torch.zeros((self.batch_size, 1))
            past_dones = torch.zeros((self.batch_size, 1))
            pred_states = states.copy()
            for step in range(self.predicted_step_n):
                pred_actions = self.v_model.get_gradient_actions(pred_states)
                pred_states, pred_rewards, pred_dones, _ = self.virtual_step_for_batch(pred_states, pred_actions)
                pred_rewards = torch.FloatTensor(pred_rewards).unsqueeze(1)
                targets += (1 - past_dones) ** step * pred_rewards
                past_dones = torch.FloatTensor(pred_dones).unsqueeze(1)
            targets += (1 - past_dones) * self.gamma * self.v_target_model(pred_states).detach()
            
            #update target model
            v_values = self.v_model(states)
            loss = torch.mean((v_values - targets.detach())**2)
            self.update_target_model(self.v_target_model, self.v_model, self.optimizer, loss)
            return None
            
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
    