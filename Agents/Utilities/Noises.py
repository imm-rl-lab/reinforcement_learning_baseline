import numpy as np


class ThreshholdNoise:
    def __init__(self, threshold=1, threshold_min=1e-2, threshold_decrease=1e-2):
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        
    def get(self):
        return None

    def reduce(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease
        return None

    def reset(self):
        return None


class DiscreteUniformNoise:
    def __init__(self, action_n, threshold=1, threshold_min=1e-2, threshold_decrease=1e-2, step_n=1):
        self.action_n = action_n
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.step_n = step_n
        self.step_i = self.step_n - 1
        self.current_action = np.random.choice(self.action_n)
            
        return None
        
    def get(self):
        if self.step_i % self.step_n == 0:
            self.step_i = self.step_n - 1
            self.current_action = np.random.choice(self.action_n)
        else:
            self.step_i -= 1
        return self.current_action

    
    def reduce(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease
        return None

    def reset(self):
        return None
    

class UniformNoise:
    def __init__(self, action_dim=1, threshold=1, threshold_min=1e-2, threshold_decrease=1e-2):
        self.action_dim = action_dim
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease

    def get(self):
        return np.random.uniform(- self.threshold, + self.threshold, self.action_dim)
    
    def reduce(self):
        if self.threshold > self.threshold_min:
            self.threshold -= self.threshold_decrease
        return None
    
    def reset(self):
        return None

    
class OUNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3, threshold=1, threshold_min=0.01,
                 threshold_decrease=0.00001, dt=1, multiplication=False):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.threshold = threshold
        self.threshold_min = threshold_min
        self.threshold_decrease = threshold_decrease
        self.multiplication = multiplication
        self.dt = dt
        self.reset()
        return None

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx * self.dt
        return self.threshold * self.state
    
    def reduce(self):
        if self.threshold > self.threshold_min:
            if self.multiplication:
                self.threshold *= self.threshold_decrease
            else:
                self.threshold -= self.threshold_decrease
        return None
        