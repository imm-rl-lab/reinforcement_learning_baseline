import os, sys
sys.path.insert(0, os.path.abspath('..'))
from Utilities.UniversalRunner import UniversalRunner


UniversalRunner('A3C_Continuous', {'env_name': ['VanDerPol'],
                                 'sn': [50],
                                 'en': [300]}, parallel=False, attempt_n=2, with_seeds=True)