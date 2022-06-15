from collections import defaultdict


def get_session(env, agent, session_len, agent_learning, use_additional_info_in_sessions=False):
    session = defaultdict(list)

    agent.reset()
    state = env.reset()
    session['states'].append(state)

    for i in range(session_len):
        action = agent.get_action(state)
        session['actions'].append(action)

        if use_additional_info_in_sessions:
            for key, value in agent.get_additional_info(state).items():
                session[key].append(value)

        next_state, reward, done, info = env.step(action)
        session['rewards'].append(reward)
        session['dones'].append(done)

        if agent_learning=='by_fives':
            agent.fit(state, action, reward, done, next_state)

        state = next_state
        session['states'].append(state)

        if done:
            session['next_value'] = 0
            break
    else:
        session['next_value'] = agent.get_value(state)


    return session


def go(env, agent, show, episode_n, session_n=1, session_len=10000, agent_learning='by_fives', use_additional_info_in_sessions=False):

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len, agent_learning, use_additional_info_in_sessions) for i in range(session_n)]

        statistics = None
        if agent_learning=='by_sessions':
            statistics = agent.fit(sessions)

        show(env, agent, episode, sessions, statistics=statistics)

    return None


def go_asynchronously(env, agent, show, episode_n, session_n=1, session_len=10000, agent_learning=True):

    for episode in range(episode_n):

        all_sessions = []
        session_array = []

        for inner_agent in agent.inner_agents:
            sessions = [get_session(env, inner_agent, session_len, agent_learning) for i in range(session_n)]
            all_sessions.append(sessions)
            session_array.extend(sessions)

        if agent_learning:
            agent.fit(all_sessions)

        show(env, agent, episode, session_array)

    return None
