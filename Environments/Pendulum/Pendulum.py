import numpy as np
import torch


class Pendulum:
    def __init__(self, initial_state=np.array([0, np.pi, 0]), dt=0.2, terminal_time=5, inner_step_n=2,
                 action_min=np.array([-2]), action_max=np.array([2])):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n

        self.gravity = 9.8
        self.r = 0.01
        self.beta = self.r
        self.m = 1.
        self.l = 1.
        self.state = self.initial_state

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n

    def g(self, state):
        return torch.stack([torch.zeros(state.shape[1]), torch.ones(state.shape[1]) * 3 / (self.m * self.l ** 2)])

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            self.state = self.state + np.array([1, self.state[2],
                                                - 3 * self.gravity / (2 * self.l) * np.sin(self.state[1] + np.pi)
                                                + 3. / (self.m * self.l ** 2) * action[0]]) * self.inner_dt

        if self.state[0] >= self.terminal_time:
            reward = - np.abs(self.state[1]) - 0.1 * np.abs(self.state[2])
            done = True
        else:
            reward = - self.r * (action[0] ** 2) * self.dt
            done = False

        return self.state, reward, done, None

    def get_state_obs(self):
        return 'time: %.3f  angle: %.3f angular velocity: %.3f' % (self.state[0], self.state[1], self.state[2])

    def render(self):
        print(self.get_state_obs())
