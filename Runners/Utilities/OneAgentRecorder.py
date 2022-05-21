import numpy as np
import pickle, os


class OneAgentRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.create_directory(directory)
        
        self.mean_total_rewards = []
        self.noise_thresholds = []
        self.best_total_rewards = []
        self.best_session = None
        
        return None
    
    def create_directory(self, directory):
        pre_directory, _ = os.path.split(directory)
        if not os.path.exists(pre_directory):
            os.mkdir(pre_directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
        return None
    
    def record_list(self, directory, data):
        with open(directory, 'wb') as file:
            pickle.dump(data, file)
        return None
    
    def record(self, env, agent, episode, sessions):
        
        total_rewards = np.array([sum(session['rewards']) for session in sessions])
        argmax_total_reward = np.argmax(total_rewards)
        mean_total_reward = np.mean(total_rewards)
        
        if self.best_session:
            if total_rewards[argmax_total_reward] > np.max(self.best_total_rewards):
                self.best_session = sessions[argmax_total_reward]  
                self.best_total_rewards.append(total_rewards[argmax_total_reward])
            else:
                self.best_total_rewards.append(np.max(self.best_total_rewards))
        else:
            self.best_session = sessions[argmax_total_reward]
            self.best_total_rewards.append(total_rewards[argmax_total_reward])
        
        self.mean_total_rewards.append(mean_total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        np.save(self.directory + '/mean_total_rewards', self.mean_total_rewards)
        np.save(self.directory + '/best_total_rewards', self.best_total_rewards)
        np.save(self.directory + '/noise_thresholds', self.noise_thresholds)
        self.record_list(self.directory + '/best_session', self.best_session)
        
        return None