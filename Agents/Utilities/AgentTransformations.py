import types
from copy import deepcopy


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

        def get_action(self, state):
            action_i = super().get_action(state)
            action = array_with_i(self.action_values[action_i])
            action.i = action_i
            return action

        def fit(self, sessions):
            for session in sessions:
                actions = []
                for action in session['actions']:
                    actions.append(action.i)
                session['actions'] = actions

            super().fit(sessions)

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


def get_asynchronous_agent(agent, inner_agent_n=10):
    inner_agents = []
    for i in range(inner_agent_n):
        inner_agents.append(deepcopy(agent))

    agent.inner_agents = inner_agents
    agent.fit = types.MethodType(new_fit, agent)

    return agent


def new_fit(self, all_sessions):
    for agent, sessions in zip(self.inner_agents, all_sessions):
        agent.fit(sessions)
        for local_pi_model, local_v_model, global_pi_model, global_v_model in zip(
                agent.pi_model.parameters(),
                agent.v_model.parameters(),
                self.pi_model.parameters(),
                self.v_model.parameters(),
        ):
            global_pi_model.grad = local_pi_model.grad
            global_v_model.grad = local_v_model.grad
            self.pi_optimizer.step()
            self.v_optimizer.step()
            agent.pi_model.load_state_dict(self.pi_model.state_dict())
            agent.v_model.load_state_dict(self.v_model.state_dict())
