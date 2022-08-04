import random
from collections import deque


class ExperienceReplayBuffer():
    def __init__(self, memory_len=100000):
        self.name = 'ExperienceReplayBuffer'
        self.values = deque(maxlen=memory_len)
        return None

    def append(self, value):
        self.values.append(value)
        return None

    def get_batch(self, batch_size):
        return random.sample(self.values, batch_size)

    def __len__(self):
        return len(self.values)
