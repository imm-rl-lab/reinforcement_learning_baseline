import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Categorical

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DiscreteActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, seed=42):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 1. убедиться, что softmax с самого начала на равномерном распределнии
        # 2. регулировань начальную инициализацию весов
        # 3. добавить blending (происходит через энтропию, но возможно, явным способом будет работать лучше)
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.critic.apply(init_weights)
        self.actor.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

    def __call__(self, x):
        dist, value = super().__call__(x)
        return dist, value
