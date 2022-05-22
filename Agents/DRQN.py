import copy
import random
import numpy as np
import torch
from collections import deque


class DRQN_PartialHistory:
    """Deep Recurrent Q-Networks algorithm for fixed history len"""

    def __init__(self, q_model, noise, history_len=4, q_model_lr=1e-3, gamma=1,
                 batch_size=32, tau=1e-3, session_memory_len=1000):

        self.q_model = q_model
        self.noise = noise
        self.history_len = history_len
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q_target_model = copy.deepcopy(self.q_model)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)

        self.session_memory = deque(maxlen=session_memory_len)
        self.new_session = True
        self.hidden_queue = []

        return None

    def get_action(self, state, noisy=True):
        state = [state]
        # reset hidden_queue
        if self.new_session:
            self.new_session = False
            self.hidden_queue = [self.get_initial_hidden(1) for _ in range(self.history_len)]
            for _ in range(self.history_len - 1):
                self.update_hidden_queue(state)

        # get q_values
        q_values, _ = self.q_model(state, self.hidden_queue[0])

        # update hidden_queue
        self.update_hidden_queue(state)

        # get action
        if noisy and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()
        else:
            return torch.argmax(q_values).item()

    # update hidden_queue
    def update_hidden_queue(self, state):
        for i in range(1, len(self.hidden_queue)):
            _, self.hidden_queue[i - 1] = self.q_model(state, self.hidden_queue[i])
        self.hidden_queue[-1] = self.get_initial_hidden(1)

    def fit(self, state, action, reward, done, next_state):
        # add to memory
        self.add_to_memory(state, action, reward, done, next_state)

        if len(self.session_memory) >= 2:
            # get history batch
            histories = self.get_history_batch()

            # calculate q_values and next_q_values
            hidden = self.get_initial_hidden(self.batch_size)
            target_hidden = self.get_initial_hidden(self.batch_size)
            for k in range(self.history_len):
                history_slice = [history[k] for history in histories]
                states, actions, rewards, danes, next_states = list(zip(*history_slice))

                q_values, hidden = self.q_model(states, hidden)
                next_q_values, target_hidden = self.q_target_model(next_states, target_hidden)

            # get targets
            targets = q_values.clone()
            for i in range(q_values.size(0)):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * torch.max(next_q_values[i])

            # train q_model
            loss = torch.mean((targets.detach() - q_values) ** 2)
            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)

        return None

    def get_initial_hidden(self, batch_size):
        return self.q_model.get_initial_state(batch_size)

    def reset(self):
        self.new_session = True
        self.noise.reset()
        self.noise.reduce()
        return None

    def add_to_memory(self, state, action, reward, done, next_state):
        if not self.session_memory or self.session_memory[-1][-1][3]:
            self.session_memory.append([[state, action, reward, done, state]] * (self.history_len - 1))

        self.session_memory[-1].append([state, action, reward, done, next_state])
        return None

    def get_history_batch(self):
        # choice sessions in accordance with their lengths
        session_weights = [len(session) - self.history_len + 1 for session in self.session_memory]
        sessions = random.choices(self.session_memory, weights=session_weights, k=self.batch_size)

        # fill history_batch
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


class DRQN_WholeHistory:
    """Deep Recurrent Q-Networks algorithm for whole history"""

    def __init__(self, q_model, noise, gamma=1, batch_size=32, burning_len=8,
                 trajectory_len=12, q_model_lr=1e-3, tau=1e-3, session_memory_len=1000):

        self.q_model = q_model
        self.noise = noise
        self.burning_len = burning_len
        self.trajectory_len = trajectory_len
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.q_target_model = copy.deepcopy(self.q_model)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)

        self.session_memory = deque(maxlen=session_memory_len)
        self.hidden = self.get_initial_state(1)

        return None

    def get_action(self, state, noisy=True):
        # get q_values
        q_values, self.hidden = self.q_model([state], self.hidden)

        # get action
        if noisy and np.random.uniform(0, 1) < self.noise.threshold:
            return self.noise.get()
        else:
            return np.argmax(q_values.data.numpy())

    def fit(self, state, action, reward, done, next_state):
        # add to memory
        self.add_to_memory(state, action, reward, done, next_state)

        if len(self.session_memory) >= 2:
            # get history batch
            trajectories = self.get_trajectories_batch()
            sliced_trajectories = [[trajectory[k] for trajectory in trajectories] for k in range(self.trajectory_len)]

            hidden = self.get_initial_state(self.batch_size)
            loss = 0
            # hidden sweep
            for k in range(self.burning_len):
                _, _, hidden, _, _, _ = self.calculate_q_values(sliced_trajectories[k], hidden)

            # detach after burning
            hidden[0].detach()
            hidden[1].detach()

            for k in range(self.burning_len, self.trajectory_len):
                # calculate q_values and next_q_values
                q_values, next_q_values, hidden, actions, rewards, danes = \
                    self.calculate_q_values(sliced_trajectories[k], hidden)

                # get targets after burning
                targets = q_values.clone()
                for i in range(self.batch_size):
                    targets[i][actions[i]] = rewards[i] + self.gamma * (1 - danes[i]) * max(next_q_values[i])
                loss += torch.mean((targets.detach() - q_values) ** 2)

            self.update_target_model(self.q_target_model, self.q_model, self.optimizer, loss)

        return None

    # calculate q_values and next_q_values
    def calculate_q_values(self, slice_trajectories, memories):
        states, actions, rewards, danes, next_states = list(zip(*slice_trajectories))

        q_values, memories = self.q_model(states, memories)
        next_q_values, _ = self.q_target_model(next_states, memories)

        return q_values, next_q_values, memories, actions, rewards, danes

    def get_initial_state(self, batch_size):
        return self.q_model.get_initial_state(batch_size)

    def reset(self):
        self.hidden = self.get_initial_state(1)
        self.noise.reset()
        self.noise.reduce()
        return None

    def add_to_memory(self, state, action, reward, done, next_state):
        if not self.session_memory or self.session_memory[-1][-1][3]:
            self.session_memory.append([])

        self.session_memory[-1].append([state, action, reward, done, next_state])
        return None

    def get_trajectories_batch(self):
        # choice sessions in accordance with their lengths
        session_weights = [len(session) - self.trajectory_len + 1 for session in self.session_memory]
        sessions = random.choices(self.session_memory, weights=session_weights, k=self.batch_size)

        # fill trajectories batch
        trajectories = []
        for session in sessions:
            start_point = random.randint(0, len(session) - self.trajectory_len)
            trajectories.append(session[start_point: start_point + self.trajectory_len])

        return trajectories

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        return None
