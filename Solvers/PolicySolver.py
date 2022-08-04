from collections import defaultdict
from copy import deepcopy

import torch


def _train_agent(env, session_len, agent):
    with torch.no_grad():
        session = defaultdict(list)
        state = env.reset()
        state = torch.FloatTensor(state)
        is_done = False
        for _ in range(session_len):
            dist, action = agent.get_dist_action(state)
            value = agent.get_value(state)

            is_done = False
            if hasattr(action, "i"):
                action_i = torch.from_numpy(action.i)
            else:
                action_i = torch.from_numpy(action)

            next_state, reward, done, _ = env.step(action)
            session['states'].append(state.cpu().numpy())
            session['actions'].append(action)
            session['rewards'].append(reward)
            session['values'].append(value.cpu().numpy()[0])
            session['log_probs'].append(
                float(dist.log_prob(action_i).cpu().numpy()))
            session['dones'].append(done)
            state = next_state
            state = torch.FloatTensor(state)
            if done:
                is_done = True
                env.reset()
        if is_done:
            session['next_value'] = 0
        else:
            next_value = agent.get_value(state)
            session['next_value'] = next_value.cpu().numpy()[0]
        return session

def go(
    env, agent, show, episode_n, session_n=1, session_len=10000,
):

    test_results = []
    envs = [deepcopy(env) for _ in range(session_n)]
    statistics = defaultdict(list)

    for episode in range(episode_n):
        sessions = [
            _train_agent(envs[i], session_len, agent)
            for i in range(session_n)
        ]
        train_statistics = agent.fit(sessions)
        show(env, agent, episode, sessions, statistics=train_statistics)
    return statistics
