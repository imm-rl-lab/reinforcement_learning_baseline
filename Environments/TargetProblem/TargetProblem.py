import numpy as np
import torch
from numpy.linalg import norm


class TargetProblem:
    def __init__(self, action_radius=np.array([1, 1]),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0]),
                 terminal_time=10, dt=0.01, inner_step_n=10, target_point=(2, 2)):

        self.state_dim = 7
        self.action_dim = 2
        self.action_radius = action_radius
        self.action_min = - self.action_radius
        self.action_max = + self.action_radius

        self.r = 0.001
        self.beta = 0.001

        self.initial_state = initial_state
        self.xG = target_point[0]
        self.yG = target_point[1]
        # уточнение орбитальной скорости вращения вокруг Земли
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.k = 1
        self.m = 1
        self.g_const = 1
        # вектор масштабирования координат состояния
        self.state = self.reset()

    def f(self, state, u):
        t, x0, y0, x, y, vx, vy = state
        ux, uy = u
        state_update = np.ones(self.state_dim)
        state_update[1] = ux
        state_update[2] = uy
        state_update[3] = vx
        state_update[4] = vy
        state_update[5] = - (self.k / self.m) * (x - x0)
        state_update[6] = - (self.k / self.m) * (y - y0) - self.g_const
        return state_update

    def g(self, state):
        t, x0, y0, x, y, vx, vy = state
        return torch.stack(
            [torch.ones(x.shape[0]), torch.ones(x.shape[0]), torch.zeros(x.shape[0]), torch.zeros(x.shape[0]),
             torch.zeros(x.shape[0]), torch.zeros(x.shape[0])])

    def reset(self):
        self.state = self.initial_state
        return self.state

    def get_state_obs(self):
        t, x0, y0, x, y, vx, vy = self.state
        return "x0=%.3f, y0=%.3f, x=%.3f, y=%.3f, vx=%.3f, vy=%.3f" % (x0, y0, x, y, vx, vy)

    def step(self, action):
        for _ in range(self.inner_step_n):
            k1 = self.f(self.state, action)
            k2 = self.f(self.state + k1 * self.inner_dt / 2, action)
            k3 = self.f(self.state + k2 * self.inner_dt / 2, action)
            k4 = self.f(self.state + k3 * self.inner_dt, action)
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

        t, x0, y0, x, y, vx, vy = self.state
        if t >= self.terminal_time:
            reward = - ((x0 ** 2) + (y0 ** 2) + ((x - self.xG) ** 2) + ((y - self.yG) ** 2))
            done = True
        else:
            reward = - self.r * (norm(action) ** 2) * self.dt
            done = False

        return self.state, reward, done, None
