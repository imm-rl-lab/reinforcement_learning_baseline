from collections import deque
from copy import deepcopy

import numpy as np
import torch
import random


class DAU():
    def __init__(self, v_model, a_model, noise, dt, an, en, v_model_lr=1e-3, a_model_lr=1e-3, gamma=1, batch_size=64,
                 v_model_tau=1e-3, a_model_tau=1e-3, memory_len=50000):
        super().__init__()
        self.v_model = v_model
        self.a_model = a_model
        self.noise = noise
        self.dt = dt
        self.gamma = gamma
        self.batch_size = batch_size
        self.v_model_tau = v_model_tau
        self.a_model_tau = a_model_tau

        self.memory = deque(maxlen=memory_len)
        self.v_target_model = deepcopy(self.v_model)
        self.a_target_model = deepcopy(self.a_model)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_model_lr)
        self.a_optimizer = torch.optim.Adam(self.a_model.parameters(), lr=a_model_lr)

        self.epsilon = 1
        self.epsilon_delta = 1 / (en - 50)
        self.action_n = an
        return None

    def get_action(self, state):
        actions = np.arange(self.action_n)
        max_action = torch.argmax(self.a_model(state))
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[max_action] += 1 - self.epsilon
        for i in range(probs.size):
            if probs[i] < 0:
                probs[i] = 0
        action = np.random.choice(actions, p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):
        """ Optimizes using the DAU variant of advantage updating.
            Note that this variant uses max_action, and not max_next_action, as is
            more common with standard Q-Learning. It relies on the set of equations
            V^*(s) + dt A^*(s, a) = r(s, a) dt + gamma^dt V^*(s)
            A^*(s, a) = adv_function(s, a) - adv_function(s, max_action)
        """

        # add to memory
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            # get batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))

            # get a values
            # A^*(s, a) = adv_function(s, a) - adv_function(s, max_action)
            adv_values = self.a_model(states)
            a_values = torch.empty(self.batch_size, 1)
            tz_rewards = torch.empty(self.batch_size, 1)
            for i in range(self.batch_size):
                tz_rewards[i][0] = rewards[i]
                max_action = torch.argmax(adv_values[i])
                a_values[i][0] = adv_values[i][actions[i]] - adv_values[i][max_action]

            # get targets
            v_values = self.v_model(states)
            next_v_values = self.v_target_model(next_states)
            q_values = (v_values + self.dt * a_values)
            targets = (tz_rewards + (self.gamma ** self.dt) * next_v_values).detach()

            # train v_model and a_model
            critic_value = (q_values - targets) ** 2
            loss = critic_value.mean()
            self.update_target_models(self.v_target_model, self.v_model, self.v_optimizer, self.a_target_model,
                                      self.a_model, self.a_optimizer, loss)
        return None

    def update_target_models(self, v_target_model, v_model, v_optimizer, a_target_model, a_model, a_optimizer, loss):
        v_optimizer.zero_grad()
        a_optimizer.zero_grad()
        loss.backward()
        v_optimizer.step()
        a_optimizer.step()
        for v_target_param, v_param in zip(v_target_model.parameters(), v_model.parameters()):
            v_target_param.data.copy_((1 - self.v_model_tau) * v_target_param.data + self.v_model_tau * v_param)
        for a_target_param, a_param in zip(a_target_model.parameters(), a_model.parameters()):
            a_target_param.data.copy_((1 - self.a_model_tau) * a_target_param.data + self.a_model_tau * a_param)
        return None

    def reset(self):
        self.noise.reset()
        self.noise.reduce()
        if self.epsilon > 0.01:
            self.epsilon -= self.epsilon_delta
        return None
