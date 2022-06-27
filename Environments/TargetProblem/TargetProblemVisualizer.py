import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class TargetProblemVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.noise_thresholds = []

    def clean(self):
        self.total_rewards = []
        self.noise_thresholds = []
        
    def show_fig(self, env, agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions], axis=0) 
                              for j in range(len(sessions[0]['actions']))])
        
        rewards = np.array([np.mean([session['rewards'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['rewards']))])
        
        plt.figure(figsize=[18, 12])

        plt.subplot(231)
        plt.plot([state[3] for state in states], [state[4] for state in states], 'm--', label='Траектория движения')
        plt.plot(env.xG,env.yG,'bo', label='Целевое состояние')
        plt.plot(states[-1][3],states[-1][4],'bo', label='Финальное состояние')
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        plt.grid()
        
        plt.subplot(232)
        plt.plot([state[1] for state in states], [state[2] for state in states], 'm--', label='Траектория движения')
        plt.plot(0,0,'bo', label='Целевое состояние')
        plt.plot(states[-1][1],states[-1][2],'bo', label='Финальное состояние')
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        plt.grid()
        
        plt.subplot(233)
        plt.plot([action[0] for action in actions], [action[1] for action in actions], 'm--', label='Управление')
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.grid()

        plt.subplot(234)
        mean_total_rewards = np.mean(self.total_rewards[-20:])
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(235)
        plt.plot(self.noise_thresholds, 'g', label='Порог шума u-агента')
        plt.legend()
        plt.grid()
        
        plt.subplot(236)
        plt.plot(rewards, 'g', label='Награды')
        plt.legend()
        plt.grid()        

        clear_output(True)
        
        plt.show()

    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.total_rewards.append(total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions)
