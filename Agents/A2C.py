import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2C():
    def __init__(self, pi_model, v_model, noise,
                 gamma=0.99, pi_model_lr=1e-3, v_model_lr=1e-3,
                 entropy_threshold=1):
        self.pi_model = pi_model
        self.v_model = v_model
        self.gamma = gamma
        self.entropy_threshold = entropy_threshold
        self.noise = noise

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_model_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), v_model_lr)
        
        return None

    #need to implement
    def get_action(self, state):
        pass

    #need to implement
    def get_dist(self, states):
        pass
    
    #train agent
    def fit(self, sessions):
        #get states, actions, and returns
        states, actions, returns = [], [], []
        for session in sessions:
            states.extend(session['states'][:-1])
            actions.extend(session['actions'])

            last_done = session['dones'][-1]
            last_state = session['states'][-1]
            last_value = self.v_model(last_state)
            current_returns = self.get_returns(session['rewards'], last_done, last_value)
            returns.extend(current_returns)   
        
        #convert actions and returns to tensor
        returns = torch.FloatTensor(returns)
        actions = self.convert_actions_to_tensor(actions)
        
        #get advantages
        advantages = returns.detach() - self.v_model(states)

        #get entropy
        dist = self.get_dist(states)
        entropy = dist.entropy().mean()

        #update pi_model
        pi_loss = - (torch.mean(dist.log_prob(actions) * advantages.detach()) + self.entropy_threshold * entropy)
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        #update v_model
        v_loss = torch.mean(advantages ** 2)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        return None

    def get_returns(self, rewards, last_done, last_value):
        returns = rewards.copy()
        returns[-1] += self.gamma * (1 - last_done) * last_value
        for i in range(len(rewards) - 2, -1, -1):
            returns[i] += self.gamma * returns[i + 1]
        return returns
    
    #need to implement
    def convert_actions_to_tensor(self, actions):
        pass
    
    def reset(self):
        return None


class A2C_Discrete(A2C):
    def __init__(self, pi_model, v_model, noise,
                 gamma=0.99, pi_model_lr=1e-3, v_model_lr=1e-3,
                 entropy_threshold=1):
        
        super().__init__(pi_model, v_model, noise, gamma, pi_model_lr, v_model_lr, entropy_threshold)
        return None

    def get_action(self, state):
        dist = self.get_dist(state)
        return int(dist.sample())

    def get_dist(self, states):
        logits = self.pi_model(states)
        probs = F.softmax(logits, -1)
        return torch.distributions.Categorical(probs)
    
    def convert_actions_to_tensor(self, actions):
        return torch.LongTensor(actions)


class A2C_Continuous(A2C):
    def __init__(self, action_min, action_max, pi_model, v_model, noise,
                 gamma=0.99, pi_model_lr=1e-3, v_model_lr=1e-3,
                 entropy_threshold=1):
        
        super().__init__(pi_model, v_model, noise, gamma, pi_model_lr, v_model_lr, entropy_threshold)
        self.action_min = action_min
        self.action_max = action_max
        return None

    def get_action(self, state):
        dist = self.get_dist(state)
        action = dist.sample().detach().data.numpy()
        return np.clip(action, self.action_min, self.action_max)
    
    def get_dist(self, states):
        mean = self.pi_model(states)
        sigma = torch.full(mean.size(), 0.1)
        return torch.distributions.Normal(mean, sigma)
    
    def convert_actions_to_tensor(self, actions):
        return torch.FloatTensor(actions)
