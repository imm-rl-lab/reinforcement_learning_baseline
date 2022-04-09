def get_session(env, agent, session_len, agent_learning):
    session = {}
    session['states'], session['actions'], session['rewards'], session['dones'] = [], [], [], []
    
    agent.reset()
    state = env.reset()
    session['states'].append(state)
    
    for i in range(session_len):
        action = agent.get_action(state)
        session['actions'].append(action)

        next_state, reward, done, info = env.step(action)
        session['rewards'].append(reward)
        session['dones'].append(done)
        
        if agent_learning=='online':
            agent.fit(state, action, reward, done, next_state)
        
        state = next_state
        session['states'].append(state)
        
        if done:
            break
        
    return session


def go(env, agent, show, episode_n, session_n=1, session_len=10000, agent_learning='offline'):

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len, agent_learning) for i in range(session_n)]

        show(env, agent, episode, sessions)
        
        #if agent_learning is 'offline':
        #    agent.fit(sessions)

    return None
