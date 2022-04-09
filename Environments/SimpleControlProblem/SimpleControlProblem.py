import numpy as np


class SimpleControlProblem:
    def __init__(self, dt=0.05, terminal_time=2, initial_state=np.array([0, 1]),
                 action_min=np.array([-1]), action_max=np.array([1])):
        self.state_dim = 2
        self.action_dim = 1
        self.dt = dt
        self.terminal_time = terminal_time
        self.initial_state = initial_state
        self.action_min = action_min
        self.action_max = action_max
        self.r = 0.5
        self.state = self.reset()
        return None

    def reset(self):
        self.state = self.initial_state 
        return self.state
    
    def dynamics(self, t, x, u):
        return x + u[0] * self.dt
    
    def cost(self, t, x, u):
        if t >= self.terminal_time:
            return - x[0] ** 2
        else:
            return - self.dt * u[0] ** 2 

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        self.state = self.state + np.array([1, action[0]]) * self.dt
        reward = - self.r * action[0] ** 2 * self.dt
        #reward = 0
        done = False
        if self.state[0] >= self.terminal_time + self.dt / 2:
            reward -= self.state[1] ** 2
            done = True
        return self.state, reward, done, None
    