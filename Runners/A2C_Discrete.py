import numpy as np
import torch.nn as nn
import argparse, torch
from time import time
import os, sys, importlib
sys.path.insert(0, os.path.abspath('..'))
from Agents.A2C import A2C_Discrete

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import DiscreteUniformNoise

from Agents.Utilities.ContinuousAgentMakers.ContinuousAgentMaker import ContinuousAgentMaker
from Agents.Utilities.Seed import seed


def run(attempt, directory, env_name, dt, an, en, sn, gamma, lrpi, lrv, ent):

    #set seed
    seed(attempt)

    #Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n = int(100 * dt))
    action_values = np.linspace(env.action_min, env.action_max, an).reshape(an, 1)

    #Agent
    noise = DiscreteUniformNoise(an, threshold_decrease=1.5/(en * sn))
    pi_model = SequentialNetwork([env.state_dim, 128, an], nn.ReLU())
    v_model = SequentialNetwork([env.state_dim, 128, 1], nn.ReLU())
    A2C = ContinuousAgentMaker(A2C_Discrete)
    agent = A2C(pi_model, v_model, noise, action_values=action_values,
                 gamma=gamma, pi_model_lr=lrpi, v_model_lr=lrv, entropy_threshold=ent)

    #Learning
    print('Start')
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=en, session_n=sn, show=recorder.record, agent_learning='by_sessions')

    print('Finish')
    return None

#def run(directory, env, dt, an, en, sn, lr, tau, pp, lipf):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default='0')
    parser.add_argument('--directory', type=str, default='../Data/A2C_Discrete')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--an', type=int, default=3)
    parser.add_argument('--en', type=int, default=100)
    parser.add_argument('--sn', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--lrpi', type=float, default=1e-2)
    parser.add_argument('--lrv', type=float, default=1e-2)
    parser.add_argument('--ent', type=float, default=0.01)
    args = parser.parse_args()
    run(args.attempt, args.directory, args.env_name, args.dt, args.an, args.en, args.sn,
        args.gamma ,args.lrpi, args.lrv, args.ent)
