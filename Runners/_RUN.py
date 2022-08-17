import os, sys
sys.path.insert(0, os.path.abspath('..'))
from Utilities.UniversalRunner import UniversalRunner


UniversalRunner('CVI', {'env_name': ['VanDerPol'], 
                        'lr':[1e-2],
                        'bs':[64],
                        'tau':[1e-2], 
                        'dt':[0.1]}, 
                        'en':[100], parallel=False, attempt_n=1)