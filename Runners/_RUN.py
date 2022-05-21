import os, sys
sys.path.insert(0, os.path.abspath('..'))
from Utilities.UniversalRunner import UniversalRunner


UniversalRunner('CEM_Discrete', {'env_name': ['SimpleControlProblem','Pendulum', 'VanDerPol', 'DubinsCar', 'TargetProblem'], 
                                 'lr':[1e-2,1e-2],
                                 'bs':[64,256],
                                 'tau':[1e-2], 
                                 'an':[3,5,9]}, parallel=False, attempt_n=5, with_seeds=True)