import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt

plt.ion()  # turn on interactive plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # decide whether to use GPU or CPU
print("Using device:", device)

env = gym.make("Humanoid-v5", render_mode="human")  # create Mujoco Humanoid environment
obs_dim = env.observation_space.shape[0]  # get state dimension
action_dim = env.action_space.shape[0]  # get action dimension
action_bound = env.action_space.high[0]  # get max value for each action

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),  # first hidden layer
            nn.ReLU(),
            nn.Linear(400, 300),  # second hidden layer
            nn.ReLU(),
            nn.Linear(300, action_dim),  # output layer
            nn.Tanh()
        )
        self.action_bound = action_bound  # store action bound

    def forward(self, x):
        return self.net(x) * self.action_bound  # output action scaled to action_bound

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),  # input is state + action
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)  # output Q-value
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))  # concatenate state and action before passing

actor = Actor(obs_dim, action_dim, action_bound).to(device)  # create actor network
target_actor = Actor(obs_dim, action_dim, action_bound).to(device)  # create target actor
critic = Critic(obs_dim, action_dim).to(device)  # create critic network
target_critic = Critic(obs_dim, action_dim).to(device)  # create target critic

target_actor.load_state_dict(actor.state_dict())  # copy weights to target actor
target_critic.load_state_dict(critic.state_dict())  # copy weights to target critic

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)  # optimizer for actor
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)  # optimizer for critic
criterion = nn.MSELoss()  # loss function for critic

memory = deque(maxlen=500000)  # replay buffer to store transitions
batch_size = 64  # mini-batch size for updates
gamma = 0.99  # discount factor
tau = 0.005  # soft update factor

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu  # initialize noise state

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu  # reset noise to mean

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)  # OU update equation
        self.state = x + dx
        return self.state

noise = OUNoise(action_dim)  # create noise process for exploration

model_path = "ddpg_humanoid.pth"
if os.path.exists(model_path):
    actor.load_state_dict(torch.load(model_path))  # load saved actor
    target_actor.load_state_dict(torch.load(model_path))  # load into target actor too
    print("Loaded model from:", model_path)

episode_rewards = []  # list to store rewards
num_episodes = 50000  # total number of episodes

fig, ax = plt.subplots()
line, = ax.plot([], [], label="Total Reward")
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Humanoid DDPG Performance (Live)')
ax.grid(True)
ax.legend()

def update_plot():
    line.set_xdata(np.arange(len(episode_rewards)))  # update X data
    line.set_ydata(episode_rewards)  # update Y data
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

for episode in range(num_episodes):
    obs, _ = env.reset()  # reset environment and get initial state
    noise.reset()  # reset noise
    episode_reward = 0
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # convert state to tensor
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()[0]  # get action from actor (without gradient)

        action = np.clip(action + noise.noise(), -action_bound, action_bound)  # add OU noise for exploration

        next_obs, reward, terminated, truncated, _ = env.step(action)  # take step in environment
        done = terminated or truncated  # check if episode finished

        memory.append((obs, action, reward, next_obs, float(done)))  # save transition to buffer
        obs = next_obs  # update current state
        episode_reward += reward  # accumulate reward

        if len(memory) >= batch_size:  # check if enough samples to update
            batch = random.sample(memory, batch_size)  # sample batch
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32).to(device)  # convert to tensors
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

            with torch.no_grad():
                next_actions = target_actor(next_states)  # get next action from target actor
                target_q = target_critic(next_states, next_actions)  # get target Q value from target critic
                y = rewards + gamma * target_q * (1 - dones)  # compute TD target

            q_val = critic(states, actions)  # compute current Q value
            critic_loss = criterion(q_val, y)  # compute critic loss (MSE)

            critic_optimizer.zero_grad()
            critic_loss.backward()  # backprop for critic
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()  # actor loss (maximize expected Q)

            actor_optimizer.zero_grad()
            actor_loss.backward()  # backprop for actor
            actor_optimizer.step()

            # soft update target critic
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # soft update target actor
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    episode_rewards.append(episode_reward)  # save episode reward

    if episode % 100 == 0:
        torch.save(actor.state_dict(), model_path)  # save actor weights every 100 episodes
        print(f"Model saved to {model_path}")

    print(f"Episode {episode}, Reward: {episode_reward:.2f}")  # log progress

    update_plot()  # update live plot

env.close()  # close environment
plt.ioff()  # turn off interactive mode
plt.savefig("ddpg_rewards_plot.png")  # save final plot
plt.show()
