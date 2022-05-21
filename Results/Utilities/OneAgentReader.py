import importlib
import os
import numpy as np
import matplotlib.pyplot as plt


class OneAgentReader():
    def __init__(self, env_name: str, show_trajectory=False):
        self.env_name = env_name
        self.show_trajectory = show_trajectory
        
        self.init_learning_fig()
        
        if self.show_trajectory:
            self.fig = getattr(importlib.import_module(env_name + 'Fig'), env_name + 'Fig')()
              
    def add_plot(self, folder: str, label: str):
        
        data = self.get_data(folder)
        sessions = [value['best_session'] for value in data.values()]
        best_total_rewards = [value['best_total_rewards'] for value in data.values()]
        mean_total_rewards = [value['mean_total_rewards'] for value in data.values()]
        self.fill_learning_fig(best_total_rewards, mean_total_rewards, label)
        
        if self.show_trajectory:
            session = self.get_mean_session(sessions)
            self.fig.fill_fig(session, label)
        
        return None
    
    def add_plots_for_all_params(self, experiment_name: str, from_i=0, to_i=None, filter_str=None):
        
        #folder = os.path.abspath(os.path.join('../../Results', experiment_name))
        folder = os.path.abspath(os.path.join('../Data', experiment_name))
        
        if to_i:
            params = os.listdir(folder)[from_i:to_i]
        else:
            params = os.listdir(folder)[from_i:]
        
        for param in params:
            if filter_str is None or filter_str in param:
                param_folder = os.path.join(folder, param)

                param = param.replace("percentile_param", "pp")
                param = param.replace("session_n", "sn")
                param = param.replace("learning_iter_per_fit", "lipf")

                self.add_plot(param_folder, param)
            

        return None
    
        
    def init_learning_fig(self):
        
        self.learning_fig = plt.figure(figsize=[12, 6])
        grid_fig = self.learning_fig.add_gridspec(1, 2)
        
        self._ax1 = self.learning_fig.add_subplot(grid_fig[0, 0], title='Best Total Rewards')
        self._ax1.grid()
        
        self._ax2 = self.learning_fig.add_subplot(grid_fig[0, 1], title='Mean Total Rewards')
        self._ax2.grid()
        
        return None
        
    def get_mean_session(self, sessions):
        session = {}
        for key in sessions[0]:
            if not key in ['done', 'info', 'total_reward']:
                session_len = min([len(session[key]) for session in sessions])
                session[key] = np.array([
                    np.mean([s[key][j] for s in sessions], axis=0) for j in range(session_len)])

        return session
    
    def get_data(self, folder: str):

        data = dict()
        for attempt in os.listdir(folder):
            folder_path = os.path.join(folder, attempt)
            folder_data = dict()
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                folder_data.update(
                    {filename.split('.')[0]: np.load(filepath, allow_pickle=True)})

            data.update({attempt: folder_data})

        return data
   

    def prepare_data(self, foldername):
        problem_folder_path = os.path.join(foldername)
        self.data = dict()
        for parameters in os.listdir(problem_folder_path):
            attempts_folder_path = os.path.join(problem_folder_path, parameters)
            attempts_data = dict()
            for attempt in os.listdir(attempts_folder_path):
                folder_path = os.path.join(attempts_folder_path, attempt)
                folder_data = dict()
                for filename in os.listdir(folder_path):
                    filepath = os.path.join(folder_path, filename)
                    folder_data.update({filename.split('.')[0]: np.load(filepath, allow_pickle=True)})
                    
                attempts_data.update({attempt: folder_data})
            
            self.data.update({parameters: attempts_data})



    def fill_learning_fig(self, best_total_rewards, mean_total_rewards, label, info=None):
        min_len = min([len(tr) for tr in best_total_rewards])
        btrs = np.array([tr[:min_len] for tr in best_total_rewards])
 
        max_btr = btrs.max(axis=0)
        min_btr = btrs.min(axis=0)
        mean_btr = btrs.mean(axis=0)
        self._ax1.fill_between(range(len(mean_btr)), min_btr, max_btr, alpha=0.1)
        self._ax1.plot(range(len(mean_btr)), mean_btr, label=label)
        self._ax1.legend()
        
        min_len = min([len(tr) for tr in mean_total_rewards])
        mtrs = np.array([tr[:min_len] for tr in mean_total_rewards])
        
        max_mtr = mtrs.max(axis=0)
        min_mtr = mtrs.min(axis=0)
        mean_mtr = mtrs.mean(axis=0)
        self._ax2.fill_between(range(len(mean_mtr)), min_mtr, max_mtr, alpha=0.1)
        self._ax2.plot(range(len(mean_mtr)), mean_mtr, label=label)
        self._ax2.legend()