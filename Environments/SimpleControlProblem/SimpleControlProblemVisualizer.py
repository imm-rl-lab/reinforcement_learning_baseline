import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class SimpleControlProblemVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.mean_total_rewards = []
        self.noise_thresholds = []

    def clean(self):
        self.mean_total_rewards = []
        self.noise_thresholds = []
        
    def show_fig(self, env, agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['actions']))])
        plt.figure(figsize=[12, 12])

        plt.subplot(221)
        plt.plot([state[0] for state in states], [state[1] for state in states], 'm--', label='Траектория движения')
        plt.plot(states[-1][0],states[-1][1],'bo', label='Финальное состояние')
        plt.xlim((0, env.terminal_time))
        plt.ylim((-2, 2))
        plt.grid()
        
        plt.subplot(222)
        plt.plot(np.arange(len(actions)) * env.dt, [action for action in actions], 'g', label='Реализация U')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(223)
        mean_total_rewards = np.mean(self.mean_total_rewards[-20:])
        label = f'mean_total_rewards: \n current={self.mean_total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
        plt.plot(self.mean_total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        if 'noise' in dir(agent):
            plt.subplot(224)
            plt.plot(self.noise_thresholds, 'g', label='Порог шума u-агента')
            plt.legend()
            plt.grid()

        clear_output(True)
        
        plt.show()

    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.mean_total_rewards.append(total_reward)
        if 'noise' in dir(agent):
            self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions)