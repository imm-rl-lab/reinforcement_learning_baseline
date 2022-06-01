import argparse
import importlib
import os
import sys

import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.abspath('..'))
from Agents.Utilities.AsynchronousAgentMaker import AsynchronousAgentMaker
from Agents.Utilities.ContinuousAgentMakers.ContinuousAgentMaker import ContinuousAgentMaker

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import DiscreteUniformNoise

from Agents.Utilities.Seed import seed


def run(attempt, directory, env_name, dt, an, en, sn, gamma, lrpi, lrv, ent, agents):

    #set seed
    seed(attempt)

    #Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n = int(100 * dt))
    action_values = np.linspace(env.action_min, env.action_max, an).reshape(an, 1)

    #Agent
    pi_model = SequentialNetwork([env.state_dim, 128, an], nn.ReLU())
    v_model = SequentialNetwork([env.state_dim, 128, 1], nn.ReLU())
    noise = DiscreteUniformNoise(an, threshold_decrease=1.5/(en * sn))
    from Agents.A2C import A2C_Discrete
    A2C_Discrete = ContinuousAgentMaker(A2C_Discrete)
    agent = A2C_Discrete(pi_model, v_model, noise, action_values=action_values, gamma=gamma,
                         pi_model_lr=lrpi, v_model_lr=lrv, entropy_threshold=ent)
    agent = AsynchronousAgentMaker(agent, agents)

    #Learning
    print('Start')
    recorder = Recorder(directory)
    solver.go_asynchronously(env, agent, episode_n=en, session_n=sn, show=recorder.record, agent_learning='by_sessions')

    print('Finish')
    return None

#def run(directory, env, dt, an, en, sn, lr, tau, pp, lipf):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default='0')
    parser.add_argument('--directory', type=str, default='../Data/A3C_Discrete')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--an', type=int, default=10)
    parser.add_argument('--en', type=int, default=100)
    parser.add_argument('--sn', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--lrpi', type=float, default=1e-2)
    parser.add_argument('--lrv', type=float, default=1e-2)
    parser.add_argument('--ent', type=float, default=0.01)
    parser.add_argument('--agents', type=int, default=10)
    args = parser.parse_args()
    run(args.attempt, args.directory, args.env_name, args.dt, args.an, args.en, args.sn,
        args.gamma, args.lrpi, args.lrv, args.ent, args.agents)
