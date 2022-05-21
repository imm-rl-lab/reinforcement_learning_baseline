import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class DubinsCarVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.noise_thresholds = []
        self.best_sessions = None
        
    def show_fig(self, env, agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['actions']))])

        plt.figure(figsize=[18, 12])
        plt.subplot(231)
        plt.plot(4,0,'bo', label='Целевое состояние')
        plt.plot(states[-1][1],states[-1][2],'bo', label='Финальное состояние')
        plt.plot([state[1] for state in states],[state[2] for state in states],'m--', label='Траектория движения')
        plt.xlim((-1,7))
        plt.ylim((-4,4))
        plt.legend(loc='upper right')
        plt.grid()
        
        plt.subplot(232)
        plt.plot(env.terminal_time,0.75 * np.pi,'bo', label='Целевое состояние')
        plt.plot([state[0] for state in states],[state[3] for state in states],'m--', label='Угол')
        plt.xlim((0, 7))
        plt.ylim((-np.pi, np.pi))
        plt.legend()
        plt.grid()        

        plt.subplot(233)
        plt.plot(np.arange(len(actions)) * env.dt, [action for action in actions],'g', label='actions')
        plt.xlim((0, env.terminal_time))
        plt.ylim((env.action_min[0] * 1.1, env.action_max[0] * 1.1))
        plt.legend()
        plt.grid()

        plt.subplot(234)
        mean_total_rewards = np.mean(self.total_rewards[-20:])
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(235)
        plt.plot(self.noise_thresholds,'g', label='noise_thrasholds')
        plt.legend()
        plt.grid()
        
        clear_output(True)
        
        #print("\t\t episode=%.0f, u_noise_threshold=%.3f, v_noise_threshold=%.3f, total reward=%.3f" %(
        #    episode, u, 0, mean_total_reward))
        #print("\t\t\t\t           final x = %.3f, final y = %.3f" %(states[-1][1], states[-1][2]))
        
        plt.show()
        
    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        if self.best_sessions is None or total_reward > max(self.total_rewards):
            self.best_sessions = sessions
        
        self.total_rewards.append(total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions)
            
    def clean(self):
        self.total_rewards = []
        self.noise_thresholds = []