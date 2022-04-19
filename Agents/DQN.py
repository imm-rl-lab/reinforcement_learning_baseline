import numpy as np
import random
import torch
from collections import deque
from copy import deepcopy


class DQN():
    def __init__(self, q_model, noise, q_model_lr=1e-3, gamma=1, 
                 batch_size=64, tau=1e-2, memory_len=50000):
        self.q_model = q_model
        self.noise = noise
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.memory = deque(maxlen=memory_len)
        self.q_target_model = deepcopy(self.q_model)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        return None

    def get_action(self, state):
        if np.random.uniform(0,1) < self.noise.threshold:
            return self.noise.get()
        else:
            qvalues = self.q_model(state).data.numpy()
            return np.argmax(qvalues)
    
    def fit(self, state, action, reward, done, next_state):
        #add to memory
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            #get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
            
            #get targets
            q_values = self.q_model(states)
            targets = q_values.clone().detach()
            next_q_values = self.q_target_model(next_states).data.numpy()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * np.max(next_q_values[i])
            
            #learn q_model
            loss = torch.mean((targets.detach() - q_values) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)

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


class DDQN(DQN):
    def __init__(self, q_model, noise, q_model_lr=1e-3, gamma=1, 
                 batch_size=64, tau=1e-2, memory_len=50000):
        '''Double DQN'''
        super().__init__(q_model, noise, q_model_lr=1e-3, gamma=1, 
                 batch_size=64, tau=1e-2, memory_len=50000)
        return None
                         
    def fit(self, state, action, reward, done, next_state):
        #add to memory
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            #get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
            
            #get targets
            q_values = self.q_model(states)
            targets = q_values.clone().detach()
            next_q_values = self.q_model(next_states)
            next_q_target_values = self.q_target_model(next_states).data.numpy()
            for i in range(self.batch_size):
                next_max_action = torch.argmax(next_q_values[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * next_q_target_values[i][next_max_action]
            
            #learn q_model
            loss = torch.mean((targets.detach() - q_values) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)

        return None
