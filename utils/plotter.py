import matplotlib.pyplot as plt
import numpy as np              

class Plotter:
    def __init__(self):
        plt.ion()  # Turn on interactive mode so the plot updates live during training
        self.fig, self.ax = plt.subplots()  # Create a figure and axes to draw on
        self.ep_rewards = []  # List to store reward for each episode
        self.avg_rewards = []  # List to store moving average rewards
        # Plot line for raw episode rewards, initially empty
        self.reward_line, = self.ax.plot([], [], label='Episode Reward')
        # Plot line for moving average rewards, initially empty, colored orange
        self.avg_line, = self.ax.plot([], [], label='Moving Avg Reward', color='orange')
        self.ax.set_xlabel('Episode')  
        self.ax.set_ylabel('Reward')   
        self.ax.set_title('DDPG Humanoid Training')  
        self.ax.legend()  
        self.ax.grid()   

    def update(self, reward, avg_reward):
        self.ep_rewards.append(reward)       # Add latest episode reward to the list
        self.avg_rewards.append(avg_reward)  # Add latest moving average to the list
        # Update x-axis data (episode numbers) and y-axis data (rewards)
        self.reward_line.set_xdata(np.arange(len(self.ep_rewards)))  # X-axis: [0, 1, 2, ..., episode count]
        self.reward_line.set_ydata(self.ep_rewards)                 # Y-axis: episode rewards
        self.avg_line.set_xdata(np.arange(len(self.avg_rewards)))  # X-axis for avg line
        self.avg_line.set_ydata(self.avg_rewards)                 # Y-axis: avg rewards
        self.ax.relim()  # Recalculate limits of plot axes
        self.ax.autoscale_view()  # Adjust view to fit new data
        plt.draw()  # Redraw the plot
        plt.pause(0.001)  # Pause a tiny moment to allow the plot to refresh visually

    def save(self, path="results/ddpg_humanoid_rewards.png"):
        plt.ioff()        # Turn off interactive mode so we can save and finalize plot
        plt.savefig(path)  # Save plot to file
        plt.show()       
