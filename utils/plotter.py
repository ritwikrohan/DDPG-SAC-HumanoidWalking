import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ep_rewards = []
        self.avg_rewards = []
        self.reward_line, = self.ax.plot([], [], label='Episode Reward')
        self.avg_line, = self.ax.plot([], [], label='Moving Avg Reward', color='orange')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('DDPG Humanoid Training')
        self.ax.legend()
        self.ax.grid()

    def update(self, reward, avg_reward):
        self.ep_rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.reward_line.set_xdata(np.arange(len(self.ep_rewards)))
        self.reward_line.set_ydata(self.ep_rewards)
        self.avg_line.set_xdata(np.arange(len(self.avg_rewards)))
        self.avg_line.set_ydata(self.avg_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    def save(self, path="results/ddpg_humanoid_rewards.png"):
        plt.ioff()
        plt.savefig(path)
        plt.show()
