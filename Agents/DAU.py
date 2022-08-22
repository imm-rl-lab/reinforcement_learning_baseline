from collections import deque
from copy import deepcopy
import numpy as np
import torch
import random


class QModel_DAU(torch.nn.Module):
    '''
    Q(state, action) = V(state) + dt * A(state, action)
    A(state, action) = a(state, action) - max a(state, action)
    '''
    def __init__(self, v_model, a_model, dt):
        super().__init__()
        self.v_model = v_model
        self.a_model = a_model
        self.dt = dt
        return None
    
    def forward(self, states):
        v_values = self.v_model(states)
        a_values = self.a_model(states)
        max_a_values = torch.amax(a_values, dim=1, keepdim=True)
        return v_values + self.dt * (a_values - max_a_values)


class DAU():
    def __init__(self, v_model, a_model, noise, dt, lr=1e-3, gamma=1, batch_size=64,
                 tau=1e-3, memory_len=1000000):
        super().__init__()
        self.q_model = QModel_DAU(v_model, a_model, dt)
        self.noise = noise
        self.dt = dt
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.memory = deque(maxlen=memory_len)
        self.q_target_model = deepcopy(self.q_model)
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)
        return None

    def get_action(self, state):
        if np.random.uniform(0,1) < self.noise.threshold:
            return self.noise.get()
        else:
            q_values = self.q_model([state]).data.numpy()[0]
            return np.argmax(q_values)

    def fit(self, state, action, reward, done, next_state):
        # add to memory
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            # get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            # get targets
            next_v_values = self.q_target_model.v_model(next_states).reshape(self.batch_size)
            targets = rewards + (1 - dones) * (self.gamma ** self.dt) * next_v_values

            # train q_model
            q_values = self.q_model(states)[torch.arange(self.batch_size), actions]
            loss = torch.mean((q_values - targets.detach()) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.q_optimizer, loss)
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
