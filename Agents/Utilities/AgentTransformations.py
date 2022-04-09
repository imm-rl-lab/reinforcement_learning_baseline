class array_with_i(list):
    'class of float number with inner attribute i'
    def __init__(self, v):
        super().__init__(v)
        self.i = None
        return None


def get_continuous_agent(DiscreteAgent):
    
    class Agent(DiscreteAgent):
        def __init__(self, *args, action_values, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_values = action_values
            
            if super().fit.__code__.co_argcount==2:
                self.fit = self.fit_by_sessions
            elif super().fit.__code__.co_argcount==6:
                self.fit = self.fit_by_fives
                
            return None

        def get_action(self, state):
            action_i = super().get_action(state)
            action = array_with_i(self.action_values[action_i])
            action.i = action_i
            return action
        
        def fit_by_sessions(self, sessions):
            for session in sessions:
                actions = []
                for action in session['actions']:
                    actions.append(action.i)
                session['actions'] = actions
            super().fit(sessions)
            return None
        
        def fit_by_fives(self, state, action, reward, done, next_action):
            super().fit(state, action.i, reward, done, next_action)
            return None
            
    return Agent


def get_continuous_agents(DiscreteAgents):
    
    class Agents(DiscreteAgents):
        def __init__(self, *args, u_action_values, v_action_values, **kwargs):
            super().__init__(*args, **kwargs)
            self.u_action_values = u_action_values
            self.v_action_values = v_action_values

        def get_u_action(self, state):
            u_action_i = super().get_u_action(state)
            u_action = array_with_i(self.u_action_values[u_action_i])
            u_action.i = u_action_i
            return u_action
        
        def get_v_action(self, state):
            v_action_i = super().get_v_action(state)
            v_action = array_with_i(self.v_action_values[v_action_i])
            v_action.i = v_action_i
            return v_action
        
        def fit(self, sessions):
            for session in sessions:
                
                u_actions = []
                for u_action in session['u_actions']:
                    u_actions.append(u_action.i)
                session['u_actions'] = u_actions
                
                v_actions = []
                for v_action in session['v_actions']:
                    v_actions.append(v_action.i)
                session['v_actions'] = v_actions
            
            super().fit(sessions)
            
            return None
            
    return Agents


def get_continuous_agents_for_every_step_learning(DiscreteAgents):
    
    class Agents(DiscreteAgents):
        def __init__(self, *args, u_action_values, v_action_values, **kwargs):
            super().__init__(*args, **kwargs)
            self.u_action_values = u_action_values
            self.v_action_values = v_action_values

        def get_u_action(self, state):
            u_action_i = super().get_u_action(state)
            u_action = array_with_i(self.u_action_values[u_action_i])
            u_action.i = u_action_i
            return u_action
        
        def get_v_action(self, state):
            v_action_i = super().get_v_action(state)
            v_action = array_with_i(self.v_action_values[v_action_i])
            v_action.i = v_action_i
            return v_action
        
        def fit(self, state, u_action, v_action, reward, done, next_state):
            super().fit(state, u_action.i, v_action.i, reward, done, next_state)
            return None
            
    return Agents