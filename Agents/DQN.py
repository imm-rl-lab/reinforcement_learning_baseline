import numpy as np
import random
import torch
from copy import deepcopy
from Agents.Utilities.Buffers.ExperienceReplayBuffer import ExperienceReplayBuffer


class DQN():
    def __init__(self, q_model, noise, q_model_lr=1e-3, gamma=1, 
                 batch_size=64, tau=1e-2, memory=ExperienceReplayBuffer(memory_len=100000)):
        self.q_model = q_model
        self.noise = noise
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.memory = memory
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
            batch = self.memory.get_batch(self.batch_size)
            states, actions, rewards, dones, next_states = list(zip(*batch))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            
            #get deltas
            q_values = self.q_model(states)[torch.arange(self.batch_size), actions]
            next_q_values = self.q_target_model(next_states).detach()
            next_v_values = torch.max(next_q_values, dim=1).values
            deltas = rewards + self.gamma * (1 - dones) * next_v_values - q_values
            
            #train q_model
            loss = torch.mean(deltas ** 2)
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
        super().__init__(q_model, noise, q_model_lr=q_model_lr, gamma=gamma, 
                 batch_size=batch_size, tau=tau, memory_len=memory_len)
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
            next_q_values = self.q_model(next_states).data.numpy()
            next_q_target_values = self.q_target_model(next_states).data.numpy()
            for i in range(self.batch_size):
                next_max_action = np.argmax(next_q_values[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * next_q_target_values[i][next_max_action]
            
            #train q_model
            loss = torch.mean((targets.detach() - q_values) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)

        return None
