#Cross-Entropy Algorithms for both discrete and continuous action spaces are implemented


import torch
import torch.optim as optim
import torch.nn.functional as func
import numpy as np
from copy import deepcopy
from Agents.Utilities.LinearTransformations import transform_interval


class CEM():
    def __init__(self, pi_model, noise, pi_model_lr, tau, percentile_param, 
                 learning_iter_per_fit):
        self.pi_model = pi_model
        self.noise = noise
        self.tau = tau
        self.percentile_param = percentile_param
        self.learning_iter_per_fit = learning_iter_per_fit
        self.optimizer = optim.Adam(params=self.pi_model.parameters(), lr=pi_model_lr)
        return None
    
    def fit(self, sessions):
        #get elite states and actions
        elite_states, elite_actions, elite_session_n = self.get_elite_states_and_actions(sessions)
        
        #learn
        if 0 < elite_session_n < len(sessions):
            for _ in range(self.learning_iter_per_fit):
                self.update_policy(elite_states, elite_actions)
        return None
    
    def get_elite_states_and_actions(self, sessions):
        #get threshold of total rewards according to percentile parameter
        total_rewards = [sum(session['rewards']) for session in sessions]
        total_reward_threshold = np.percentile(total_rewards, self.percentile_param)
        
        #get elite states and actions
        elite_states, elite_actions = [], []
        elite_session_n = 0
        for session in sessions:
            if sum(session['rewards']) >= total_reward_threshold:
                session_len = min(len(session['states']), len(session['actions']))
                elite_states.extend(session['states'][:session_len])
                elite_actions.extend(session['actions'][:session_len])
                elite_session_n += 1

        return elite_states, elite_actions, elite_session_n
    
    def update_model(self, model, optimizer, loss):
        #gradient step
        copy_pi_midel = deepcopy(self.pi_model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #soft update
        for param, copy_param in zip(self.pi_model.parameters(), copy_pi_midel.parameters()):
                param.data.copy_(self.tau * param + (1 - self.tau) * copy_param)
        return None
    
    def reset(self):
        self.noise.reset()
        self.noise.reduce()
        return None


class CEM_Discrete(CEM):
    def __init__(self, pi_model, noise, pi_model_lr, tau=1e-2, percentile_param=70, 
                 learning_iter_per_fit=16, greedy_policies=False):

        super().__init__(pi_model, noise, pi_model_lr, tau, 
                         percentile_param, learning_iter_per_fit)
        
        self.noise = noise
        self.greedy_policies = greedy_policies
        return None
        
    def get_action(self, state):
        if np.random.uniform() < self.noise.threshold:
            return self.noise.get()
        else:
            probs = self.get_probs(self.pi_model, state)
            if self.greedy_policies:
                action = np.argmax(probs.detach().data.numpy())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample())
            return action
    
    def get_probs(self, model, state):
        return func.softmax(model(state), -1)
    
    def update_policy(self, elite_states, elite_actions):
        #get loss
        logits = self.pi_model(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        loss = func.cross_entropy(logits, elite_actions)
        
        #learn
        self.update_model(self.pi_model, self.optimizer, loss)
        return None
    
class CEM_Continuous(CEM):
    
    def __init__(self, action_min, action_max, pi_model, noise, 
                 pi_model_lr=1e-3, tau=1e-2, percentile_param=70, learning_iter_per_fit=16):
        
        super().__init__(pi_model, noise, pi_model_lr, tau, 
                         percentile_param, learning_iter_per_fit)
        
        self.action_min = action_min
        self.action_max = action_max
        return None
        
    def get_action(self, state):
        action = self.pi_model(state).detach().numpy() + self.noise.get()
        action = self.transform_interval(action)
        return np.clip(action, self.action_min, self.action_max)
    
    def transform_interval(self, action):
        return transform_interval(action, self.action_min, self.action_max)
    
    def update_policy(self, elite_states, elite_actions):
        #get loss
        elite_actions = torch.FloatTensor(elite_actions)
        pred_actions = self.transform_interval(self.pi_model(elite_states))
        loss = torch.mean((pred_actions - elite_actions) ** 2)
        
        #learn
        self.update_model(self.pi_model, self.optimizer, loss)
        return None
    
    