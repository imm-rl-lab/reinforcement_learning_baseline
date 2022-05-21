# Finite-Horizon Control Problems

We tends to realize stable versions of popular deep reinforcement learning algorithms and test their in Finite-Horizon Control Problems. Since all sessions have the same lengths in such problems, we find it reasonable to separate the processes of exploitation and learning. In other words, an agent does not learn on every steps, but it alternates an experience accumulation during several session and a multi-step learning.

The following algorithms are considered:

**Cross-Entropy Method (CEM)**
- paper: [The Differentiable Cross-Entropy Method](https://arxiv.org/pdf/1909.12830.pdf)
- implementation: Ivan Pavelyev, Anton Plaksin

**Deep Q-Network (DQN)**
- paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

**Double Deep Q-Network (DDQN)**
- paper: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)

**Deep Deterministic Policy Gradient (DDPG)**
- paper: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)

**Normalized Adavtage Functions (NAF)**
- paper: [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748.pdf)
- implementation: Stepan Martyanov

**Asynchronous Advantage Actor-Critic (A3C)**
- paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
- implementation: Alexander Chaikov
