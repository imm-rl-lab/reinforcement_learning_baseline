import argparse
import importlib
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from Agents.DRQN import DRQN_WholeHistory
from Agents.Utilities.ContinuousAgentMakers.ContinuousAgentMaker import ContinuousAgentMaker
from Agents.Utilities.Noises import DiscreteUniformNoise
from Agents.Utilities.SequentialNetworkWithTypes import SequentialNetworkWithTypes, LayerType

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.Seed import seed


def run(attempt, directory, env_name, dt, an, lr, en, gamma, bs, bl, tl, ml, tau):
    # set seed
    seed(attempt)

    # Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n=int(100 * dt))

    # Agent
    action_values = np.linspace(env.action_min, env.action_max, an).reshape(an, 1)
    DRQN = ContinuousAgentMaker(DRQN_WholeHistory)
    q_model = SequentialNetworkWithTypes(env.state_dim,
                                         [(LayerType.Dense, 128),
                                          (LayerType.LSTM, 64),
                                          (LayerType.Dense, 64),
                                          (LayerType.Dense, an)])
    noise = DiscreteUniformNoise(an, threshold_decrease=1.5 / en)
    agent = DRQN(q_model, noise, action_values=action_values, burning_len=bl, trajectory_len=tl, q_model_lr=lr,
                 gamma=gamma, batch_size=bs, session_memory_len=ml, tau=tau)

    # Learning
    print('Start')
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=en, show=recorder.record)

    # Finish
    print('Finish')
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default='0')
    parser.add_argument('--directory', type=str, default='0')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--an', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--en', type=int, default=100)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--bl', type=int, default=4)
    parser.add_argument('--ml', type=int, default=1000)
    parser.add_argument('--tl', type=int, default=8)
    args = parser.parse_args()
    run(args.attempt, args.directory, args.env_name, args.dt, args.an, args.lr, args.en, args.gamma, args.bs, args.bl,
        args.tl, args.ml, args.tau)
