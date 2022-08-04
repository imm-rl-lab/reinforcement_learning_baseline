import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

from typing import Any
from Agents.Utilities.AdvantageEstimators import get_gae_returns


class PPO:
    def __init__(self,
                 pi_model: nn.Module,
                 v_model: nn.Module,
                 batch_size: int,
                 epochs: int,
                 pi_model_lr: float = 1e-3,
                 v_model_lr: float = 1e-3,
                 pi_model_max_grad_norm: float = 1,
                 v_model_max_grad_norm: float = 1,
                 gamma: float = 0.95,
                 lambda_: float = 0.99,
                 clip_epsilon: float = 0.2,
                 value_loss_coefficient: float = 0.5,
                 entropy_bonus_coefficient: float = 1e-3,
                 random_state=42):
        """
            Нейронные сети и параметры обучения:

            pi_model, v_model:              модели, которая описывают функции: (state) -> (distribution), (state) -> (value)
            batch_size:                     размер выборки из получившихся actors * timestamps запусков
            epochs:                         количество раз для взятия batch из actors * timestamps запусков
            pi_model_lr:                    learning rate для Adam оптимизатора для pi_model
            v_model_lr:                     learning rate для Adam оптимизатора для v_model
            pi_model_max_grad_norm:         Ограничение на норму градиента модели при обучении для pi_model
            v_model_max_grad_norm:          Ограничение на норму градиента модели при обучении для v_model

            Параметры для оценки Advantage функции:

            gamma:                          параметр для суммирования наград с затуханием
            lambda_:                        параметр для gae

            Параметры для loss функции:

            clip_epsilon:                   Коэффициент с которым происходит обрезание отношения новой и старой политики
            value_loss_coefficient:         коэффициент перед Squared error для value функции.
                Так как мы используем AC модель, то нам нужно учитывать как ошибки для суррогатной функции потерь и ошибки value функции. Для этого определяем данный коэффициент.
            entropy_bonus_coefficient:      Коэффициент перед энтропией распределения. Нужно для того, чтобы обеспечить достаточное количество исследования различных траекторий на этапе обучения.
        """
        self.pi_model = pi_model
        self.v_model = v_model

        self.batch_size = batch_size
        self.epochs = epochs

        self.pi_optimizer = optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.v_optimizer = optim.Adam(self.v_model.parameters(), lr=v_model_lr)

        self.pi_model_max_grad_norm = pi_model_max_grad_norm
        self.v_model_max_grad_norm = v_model_max_grad_norm

        self.gamma = gamma
        self.lambda_ = lambda_

        self.clip_epsilon = clip_epsilon
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_bonus_coefficient = entropy_bonus_coefficient

        self.rng = np.random.default_rng(seed=random_state)

    @abc.abstractmethod
    def get_action(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dist(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dist_action(self, state):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_value(self, state):
        raise NotImplementedError()

    def get_pi_loss(self, states, actions, advantages, old_log_probs, statistics=None) -> torch.FloatTensor:
        """
            Рассчёт функций потерь для pi_model:

            states:             Состояния в сессии
            actions:            Действия в сессии
            advantages:         Оценки advantage функции по сессии
            old_log_probs:      Логарифмф вероятности действий в сессии

            returns:            Тензор со значением функции потерь для pi_model
        """
        dist = self.get_dist(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()
        surrogate_function_1 = ratio * advantages
        surrogate_function_2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss  = torch.min(surrogate_function_1, surrogate_function_2).mean()

        pi_model_loss = - policy_loss - self.entropy_bonus_coefficient * entropy

        if statistics:
            statistics.update(entropy=entropy.detach().numpy())

        return pi_model_loss

    def get_v_loss(self, states, values_targets, values) -> torch.FloatTensor:
        """
            Рассчёт функций потерь для v_model:

            states:             Состояния в сессии
            values_targets:     Оценки value для сессии
            values:             Значения value модели v_model для сесии

            returns:            Тензор со значением функции потерь для v_model
        """
        current_values = self.get_value(states)
        l_simple = (current_values - values_targets) ** 2
        v_diff_clipped = torch.clamp(current_values - values, -self.clip_epsilon, self.clip_epsilon)
        l_clipped = (values + v_diff_clipped - values_targets) ** 2
        values_loss = torch.max(l_simple, l_clipped).mean()

        v_model_loss = self.value_loss_coefficient * values_loss

        return v_model_loss

    def get_memory_buffer_from_sessions(self, sessions: list[dict[str, list[Any]]], statistics: dict[str, Any]) -> list[list[Any]]:
        states, actions, advantages, values_targets, values, log_probs = [], [], [], [], [], []
        for session in sessions:
            states.extend(session['states'])
            actions.extend(session['actions'])
            log_probs.extend(session['log_probs'])
            values.extend(session['values'])

            _advantages, _values_targets = get_gae_returns(
                session['rewards'],
                session['values'],
                session['next_value'],
                session['dones'],
                gamma=self.gamma,
                lambda_=self.lambda_
            )
            advantages.extend(_advantages)
            values_targets.extend(_values_targets)
        statistics.update(
            r2_score=r2_score(values_targets, values)
        )
        return np.array(list(zip(states, actions, advantages, values_targets, values, log_probs)), dtype=object)

    def fit(self, sessions: list[dict[str, list[Any]]]) -> None:
        """
            Основной цикл обучения агента:

            sessions: Список сессий для агентов со взаимодействием со средой
        """
        statistics = dict()
        memory = self.get_memory_buffer_from_sessions(sessions, statistics)

        for _ in range(self.epochs):
            # 1. дать пятёркам метки с вероятностью и семплировать батчи согласно им
            batch = (
                self.rng.choice(memory, size=self.batch_size)
                if self.batch_size < len(memory)
                else self.rng.shuffle(memory)
            )
            batch_states, batch_actions, batch_advantages, batch_values_targets, batch_values, batch_log_probs = zip(*batch)

            batch_states            = torch.from_numpy(np.stack(batch_states)).float()
            batch_actions           = torch.from_numpy(np.stack(batch_actions)).float()
            batch_advantages        = torch.tensor(batch_advantages).float()
            batch_values_targets    = torch.tensor(batch_values_targets).float()
            batch_values            = torch.tensor(batch_values).float()
            batch_log_probs         = torch.tensor(batch_log_probs).float()

            self.pi_optimizer.zero_grad()
            pi_loss = self.get_pi_loss(batch_states, batch_actions, batch_advantages, batch_log_probs, statistics)
            pi_loss.backward()
            pi_grad_norm = nn.utils.clip_grad_norm_(self.pi_model.parameters(), self.pi_model_max_grad_norm)
            self.pi_optimizer.step()

            self.v_optimizer.zero_grad()
            v_loss = self.get_v_loss(batch_states, batch_values_targets, batch_values)
            v_loss.backward()
            v_grad_norm = nn.utils.clip_grad_norm_(self.v_model.parameters(), self.v_model_max_grad_norm)
            self.v_optimizer.step()

            statistics.update(
                pi_loss=pi_loss.detach().numpy(),
                pi_grad_norm=pi_grad_norm.detach().numpy(),
                v_loss=v_loss.detach().numpy(),
                v_grad_norm=v_grad_norm.detach().numpy(),
                advantages=batch_advantages.numpy(),
                values_targets=batch_values_targets.numpy(),
                values=batch_values.numpy()
            )

        return statistics

    def reset(self):
        pass

class PPO_Continious(PPO):
    def __init__(self, *args, continious_scheme, **kwargs):
        super().__init__(*args, **kwargs)
        self.continious_scheme = continious_scheme

    def get_dist_action(self, state):
        state = torch.FloatTensor(state)
        dist = self.get_dist(state)
        action = dist.sample().detach().numpy()
        return dist, action

    def get_value(self, state):
        return self.v_model(state)

    def get_dist(self, state):
        logits = self.pi_model(state)
        return self.continious_scheme(logits)


class PPO_Discrete(PPO_Continious):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, continious_scheme=PPO_Discrete._discrete_scheme, **kwargs)

    @staticmethod
    def _discrete_scheme(logits):
        probs = F.softmax(logits, -1)
        return torch.distributions.Categorical(probs)
