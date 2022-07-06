import argparse
import importlib
import os
import sys

import numpy as np

from Agents.Utilities.Noises import DiscreteUniformNoise
from Agents.Utilities.SequentialNetwork import SequentialNetwork

sys.path.insert(0, os.path.abspath('..'))

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.Seed import seed
from Agents.DAU import DAU
from Agents.Utilities.ContinuousAgentMakers.ContinuousAgentMaker import ContinuousAgentMaker


def run(attempt, directory, env_name, dt, an, en, v_lr, a_lr, gamma, bs, v_tau, a_tau, ml):
    # set seed
    seed(attempt)

    # Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n=int(100 * dt))

    # Agent
    action_values = np.linspace(env.action_min, env.action_max, an).reshape(an, 1)
    dau = ContinuousAgentMaker(DAU)
    v_model = SequentialNetwork([env.state_dim, 256, 128, 1])
    a_model = SequentialNetwork([env.state_dim, 256, 128, an])
    noise = DiscreteUniformNoise(an, threshold_decrease=1.2 / en)
    agent = dau(v_model, a_model, noise, dt, an, en, action_values=action_values, v_model_lr=v_lr, a_model_lr=a_lr,
                gamma=gamma, batch_size=bs, v_model_tau=v_tau, a_model_tau=a_tau, memory_len=ml)

    # Learning
    print(f"Start attempt {attempt}")
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=en, show=recorder.record)

    # Finish
    print(f'Finish attempt {attempt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--directory', type=str, default='0')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--an', type=int, default=3)
    parser.add_argument('--en', type=int, default=200)
    parser.add_argument('--v_lr', type=float, default=1e-3)
    parser.add_argument('--a_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--v_tau', type=float, default=1e-3)
    parser.add_argument('--a_tau', type=float, default=1e-3)
    parser.add_argument('--ml', type=int, default=50000)
    args = parser.parse_args()
    run(args.attempt, args.directory, args.env_name, args.dt, args.an, args.en, args.v_lr, args.a_lr,
        args.gamma, args.bs, args.v_tau, args.a_tau, args.ml)
