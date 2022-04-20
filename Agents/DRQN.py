import copy
import random
import numpy as np
import torch
from collections import deque


class DRQN_PartialHistory:
    '''Deep Recurrent Q-Networks algorithm for fixed history len'''

    def __init__(self, q_modal, noise, history_len=4, q_model_lr=1e-3, gamma=1, 
                 batch_size=32, tau=1e-3, session_memory_len=1000):

        self.q_model = q_modal
        self.noise = noise
        self.history_len = history_len
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory_size = memory_size
        
        self.q_target_model = copy.deepcopy(self.q_model)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)

        self.session_memory = deque(maxlen=session_memory_len)
        self.previous_done = False
        self.hidden_queue = []
        self.reset()
        
        return None

    def get_action(self, state):
        #get q_values
        q_values, _ = self.q_model(state, self.hidden_queue[0])

        #update hidden_queue
        for i in range(1, len(self.hidden_queue)):
            _, self.hidden_queue[i - 1] = self.q_model(state, self.hidden_queue[i])
        self.hidden_queue[-1] = self.get_initial_hiddens(1)

        #get action
        if np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()
        else:
            return np.argmax(q_values.data.numpy())

    def fit(self, state, action, reward, done, next_state):
        #add to memory
        self.add_to_memory(state, action, reward, done, next_state)

        if len(self.session_memory) >= self.batch_size:
            #get history batch
            histories = self.get_history_batch()

            #calculate q_values and next_q_values
            hiddens = self.get_initial_hiddens(self.batch_size)
            target_hiddens = self.get_initial_hiddens(self.batch_size)
            for k in range(self.history_len):
                history_slice = [history[k] for history in histories]
                states, actions, rewards, danes, next_states = list(zip(*history_slice))

                q_values, hiddens = self.q_model(states, hiddens)
                next_q_values, target_hiddens = self.q_target_model(next_states, target_hiddens)

            #get targets
            targets = q_values.clone()
            for i in range(q_values.size(0)):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])
            
            #train q_model
            loss = torch.mean((targets.detach() - q_values) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)
            
        return None

    def get_initial_hiddens(self, batch_size):
        return self.q_model.get_initial_state(batch_size)

    def reset(self):
        self.hidden_queue = [self.get_initial_hiddens(1) for _ in range(self.history_len)]

    def add_to_memory(self, state, action, reward, done, next_state):
        
        if self.previous_done:
            self.session_memory.append([[state, action, reward, done, state]] * (self.history_len - 1))
        self.previous_done =  done
               
        self.session_memory[-1].append([state, action, reward, done, next_state])
        return None
        
    def get_history_batch(self):
        #choice sessions in accordance with their lengths
        session_weights = [len(session) - self.history_len + 1 for session in self.session_memory]
        sessions = random.choices(self.session_memory, weights=session_weights, k=self.batch_size)

        #fill history_batch
        histories = []
        for session in sessions:
            start_point = random.randint(0, len(session) - self.history_len)
            histories.append(session[start_point: start_point + self.history_len])

        return histories
    
    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None


#class DRQN_WholeHistory:
# ...