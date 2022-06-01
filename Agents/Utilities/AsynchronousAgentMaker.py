import types
from copy import deepcopy


def AsynchronousAgentMaker(agent, inner_agent_n=10):
    inner_agents = []
    for i in range(inner_agent_n):
        inner_agents.append(deepcopy(agent))

    agent.inner_agents = inner_agents
    agent.fit = types.MethodType(asynchronous_fit, agent)

    return agent


def asynchronous_fit(self, all_sessions):
    for agent, sessions in zip(self.inner_agents, all_sessions):
        agent.fit(sessions)
        self.pi_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
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
        
    return None
