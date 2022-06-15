from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output


class SimpleControlProblemVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.mean_total_rewards = []
        self.noise_thresholds = []
        self.statistics = defaultdict(list)

    def clean(self):
        self.mean_total_rewards = []
        self.noise_thresholds = []

    def update_statistics(self, train_statistics=None):
        if not train_statistics:
            return
        for key, values in train_statistics.items():
            self.statistics[key].append(values)

    def show_fig(self, env, agent, sessions, statistics):
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0)
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions])
                              for j in range(len(sessions[0]['actions']))])

        def show_plot():
            # show trajectory
            ax = yield True
            ax.plot([state[0] for state in states], [state[1] for state in states], 'm--', label='Траектория движения')
            ax.plot(states[-1][0],states[-1][1],'bo', label='Финальное состояние')
            ax.set_xlim((0, env.terminal_time))
            ax.set_ylim((-2, 2))
            ax.grid()

            # show U realization
            ax = yield True
            ax.plot(np.arange(len(actions)) * env.dt, [action for action in actions], 'g', label='Реализация U')
            ax.set_xlim((0, env.terminal_time))
            ax.legend()
            ax.grid()

            # show mean total reward
            mean_total_rewards = np.mean(self.mean_total_rewards[-20:])
            label = f'mean_total_rewards: \n current={self.mean_total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
            ax = yield True
            ax.plot(self.mean_total_rewards, 'g', label=label)
            ax.legend()
            ax.grid()

            # show noise
            if 'noise' in dir(agent):
                ax = yield True
                ax.plot(self.noise_thresholds, 'g', label='Порог шума u-агента')
                ax.legend()
                ax.grid()

            # show statistics
            for key, value in self.statistics.items():
                ax = yield True
                ax.plot(value)
                last = value[-1]
                if not isinstance(last, np.ndarray) or len(last.shape) == 0:
                    ax.set_title(f"{key}. Last: {last:.2f}")
                else:
                    ax.set_title(key)
                ax.grid()
            yield False

        width = 2
        height = (4 + len(self.statistics.keys()) + (width - 1)) // width

        fig = plt.figure(figsize=(7 * width, 3 * height))
        gs = gridspec.GridSpec(height, width)
        np.set_printoptions(precision=2)

        show_gen = show_plot()
        next(show_gen)

        for i in range(1, width * height + 1):
            ax = fig.add_subplot(gs[i // width, i % width])
            if not show_gen.send(ax):
                break

        clear_output(True)

        plt.subplots_adjust(wspace=.1, hspace=.3)
        plt.show()

    def show(self, env, agent, episode, sessions, statistics=None):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        self.update_statistics(statistics)

        self.mean_total_rewards.append(total_reward)
        if 'noise' in dir(agent):
            self.noise_thresholds.append(agent.noise.threshold)

        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions, statistics=statistics)
