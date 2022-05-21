import numpy as np
import torch.nn as nn
import argparse, torch
from time import time
import os, sys
sys.path.insert(0, os.path.abspath('..'))

from Solvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder
from Agents.Utilities.Seed import seed

#YOUR IMPORTS HERE

def run(directory, env, dt, lr, en):
    
    #set seed
    seed(attempt)
    
    #Environments
    env_path = 'Environments.' + env_name + '.' + env_name
    env = getattr(importlib.import_module(env_path), env_name)(dt=dt, inner_step_n = int(100 * dt))

    #Agent
    agent = #YOUR AGENT HERE

    #Learning
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=episode_n, show=recorder.record)
    
    #Recording
    #torch.save(pi_model, directory + '/pi_model.pt')
    
    #Finish
    print('Finish')
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='0')
    parser.add_argument('--env_name', type=str, default='SimpleControlProblem')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--en', type=int, default=100)
    args = parser.parse_args()
    run(args.directory, args.env_name, args.dt, args.lr, args.en)