# Finite-Horizon Control Problems

We tends to realize stable versions of popular deep reinforcement learning algorithms and test their in Finite-Horizon Control Problems. Since all sessions have the same lengths in such problems, we find it reasonable to separate the processes of exploitation and learning. In other words, an agent does not learn on every steps, but it alternates an experience accumulation during several session and a multi-step learning.

The following algorithms are considered:

*Continuous action space:*

- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Normalized Adavtage Functions (NAF)](https://arxiv.org/pdf/1603.00748.pdf)

*Discrete action space:*

- [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602.pdf)
- [Double Deep Q-Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)

