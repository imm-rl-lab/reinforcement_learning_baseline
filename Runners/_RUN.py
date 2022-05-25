import os, sys
sys.path.insert(0, os.path.abspath('..'))
from Utilities.UniversalRunner import UniversalRunner


UniversalRunner('A2C_Discrete', {'env_name': ['SimpleControlProblem'], 
                                 'lrpi':[1e-2],
                                 'an':[5]}, parallel=True, attempt_n=3, with_seeds=True)