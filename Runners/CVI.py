import numpy as np
import torch.nn as nn
import argparse, torch
from time import time
import os, sys, importlib
sys.path.insert(0, os.path.abspath('..'))

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import OUNoise
from Agents.Utilities.ContinuousAgentMakers.ContinuousAgentMaker import ContinuousAgentMaker
from Agents.Utilities.Seed import seed
from Agents.CVI import CVI, VModelWithGradient


def run(attempt, directory, env_name, dt, en, lr, tau, bs, psn):
    
    #set seed
    seed(attempt)
    
    #Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n = int(100 * dt))

    #Agent
    v_model_backbone = SequentialNetwork([env.state_dim, 256, 128, 1], nn.ReLU())
    v_model = VModelWithGradient(env.action_min, env.action_max, v_model_backbone, env.g, env.r)
    noise = OUNoise(action_dim=env.action_dim, threshold_decrease=1/en)
    agent = CVI(env.action_min, env.action_max, v_model, virtual_step=env.virtual_step_for_batch, noise=noise,
                batch_size=bs, gamma=1, tau=tau, v_model_lr=lr, predicted_step_n=psn)

    #Learning
    print('Start')
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=en, show=recorder.record)
    
    print('Finish')
    return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default='0')
    parser.add_argument('--directory', type=str, default='../Data/CVI')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--en', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--psn', type=int, default=4)
    args = parser.parse_args()
    run(args.attempt, args.directory, args.env_name, args.dt, args.en, args.lr, args.tau, args.bs, args.psn)
