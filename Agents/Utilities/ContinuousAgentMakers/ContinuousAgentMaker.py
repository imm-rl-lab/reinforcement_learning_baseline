class array_with_i(list):
    'class of float number with inner attribute i'
    def __init__(self, v):
        super().__init__(v)
        self.i = None
        return None


def ContinuousAgentMaker(DiscreteAgent):
    '''The class convert an agent for a discrete action space 
    to the agent for the continuous action space'''

    class ContinuousAgent(DiscreteAgent):
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

    return ContinuousAgent
