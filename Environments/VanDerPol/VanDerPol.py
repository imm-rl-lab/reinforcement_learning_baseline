import numpy as np
import torch


class VanDerPol:
    def __init__(self, initial_state=np.array([0,1,0]), action_min=np.array([-1]), action_max=np.array([+1]), 
                 terminal_time=11, dt=0.1, inner_step_n=10):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.initial_state = initial_state
        self.state = self.initial_state
        self.r = 0.05
        return None
    
    def reset(self):
        self.state = self.initial_state
        return self.state
    
    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        
        for _ in range(self.inner_step_n):
            f = np.array([1, self.state[2], (1 - self.state[1] ** 2) * self.state[2] - self.state[1] + action[0]])
            self.state = self.state + f * self.inner_dt
            
        if self.state[0] < self.terminal_time:
            done = False
            reward = - self.r * action[0] ** 2 * self.dt
        else:
            done = True
            reward = - self.state[1] ** 2 - self.state[2] ** 2
        
        return self.state, reward, done, _

    def virtual_step_for_batch(self, states, actions):
        actions = np.clip(actions, self.action_min, self.action_max)
        
        for _ in range(self.inner_step_n):
            dynamic = np.column_stack([np.ones(states.shape[0]), states[:, 2], 
                                       (1 - states[:, 1] ** 2) * states[:, 2] - states[:, 1] + actions[:, 0]])
            states = states + dynamic * self.inner_dt

        dones = np.full(states.shape[0], False)
        dones[states[:, 0] >= self.terminal_time - self.dt / 2] = True
            
        rewards = - self.r * actions[:, 0] ** 2 * self.dt
        rewards[dones] = -states[dones, 1] ** 2 - states[dones, 2] ** 2
        
        return states, rewards, dones, _
    
    def g(self, state):
        return np.array([[0, 1]])
