import numpy as np
import random
from collections import deque


class PrioritizedExperienceReplayBuffer():
    def __init__(self, alpha=0.7, epsilon=1e-6, memory_len=100000):
        self.name = 'PrioritizedExperienceReplayBuffer'
        self.alpha = alpha
        self.epsilon = epsilon
        self.values = deque(maxlen=memory_len)
        self.priorities = deque(maxlen=memory_len)
        self.index_batch = None
        return None

    def append(self, value):
        priority = max(self.priorities) ** self.alpha if len(self) > 0 else 1
        self.priorities.append(priority)
        self.values.append(value)
        return None
        
    def get_batch(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        self.index_batch = np.random.choice(np.arange(len(self)), size=batch_size, p=probs)
        return [self.values[i] for i in self.index_batch]
    
    def update_priorities(self, delta_batch):
        for index, delta in zip(self.index_batch, delta_batch):
            self.priorities[index] = (abs(delta) + self.epsilon) ** self.alpha
        return None
    
    def get_weights(self, beta=0.5):
        priority_batch = np.array([self.priorities[i] for i in self.index_batch])
        return (min(self.priorities) / priority_batch) ** beta
    
    def __len__(self):
        return len(self.values)
