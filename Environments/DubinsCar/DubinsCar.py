import numpy as np


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=10,
                 action_min=np.array([-0.5]), action_max=np.array([1]), action_n=None):
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
        self.action_n = action_n
        self.r = 0.01
        
        if self.action_n:
            self.action_values = np.linspace(self.action_min, self.action_max, self.action_n).reshape(self.action_n, 1)

        return None

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        
        for _ in range(self.inner_step_n):
            self.state = self.state + np.array([1, np.cos(self.state[3]), np.sin(self.state[3]), action[0]]) * self.inner_dt

        reward = - self.r * (action[0] ** 2) * self.dt
        done = False
        if self.state[0] >= self.terminal_time:
            reward -= np.abs(self.state[1] - 4) + np.abs(self.state[2]) + np.abs(self.state[3] - 0.75 * np.pi)
            done = True

        return self.state, reward, done, None
    
    def virtual_step_for_batch(self, states, actions):
        actions_raw = actions.copy()
        actions = np.clip(actions, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            dynamic = np.column_stack([np.ones(states.shape[0]), np.cos(states[:, 3]), np.sin(states[:, 3]), actions[:, 0]])
            states = states + dynamic * self.inner_dt

        dones = np.full(states.shape[0], False)
        dones[states[:, 0] >= self.terminal_time - self.dt / 2] = True

        rewards = - self.r * actions[:, 0] ** 2 * self.dt
        rewards[dones] = -np.abs(states[dones, 1] - 4) - np.abs(states[dones, 2]) - np.abs(states[dones, 3] - 0.75 * np.pi)
        
        return states, rewards, dones, None
    
    def g(self, g):
        return np.array([[0], [0], [1]])
    
    
    def dynamics(self, t, x, u):
        return x + np.array([np.cos(x[2]), np.sin(x[2]), u[0]]) * self.dt
    
    def cost(self, t, x, u):
        if t >= self.terminal_time:
            return - np.abs(x[0] - 4) ** 2 - np.abs(x[1]) ** 2 - np.abs(x[2] - 0.75 * np.pi) ** 2
        else:
            return - self.r * (u[0] ** 2) * self.dt
