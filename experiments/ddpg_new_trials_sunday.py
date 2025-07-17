import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# Hyperparameters
ENV_NAME = "Humanoid-v5"
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.995
TAU = 0.001
BATCH_SIZE = 256
MEMORY_SIZE = 1_000_000
EPISODES = 20000
MAX_STEPS = 1000
START_TRAIN_AFTER = 5000
SAVE_INTERVAL = 500
NOISE_SIGMA = 0.3
NOISE_THETA = 0.15
NOISE_DECAY = 0.995
SIGMA_MIN = 0.05
REWARD_SCALE = 1.0
MODEL_PATH = "ddpg_humanoid_final_sunday_new_trial.pth"
MOVING_AVG_WINDOW = 50

env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)  # I use deque to store transitions

    def put(self, transition):
        self.buffer.append(transition)  # I add new transition

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # I randomly sample a batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device))

    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),  # first hidden layer
            nn.ReLU(),
            nn.Linear(400, 300),  # second hidden layer
            nn.ReLU(),
            nn.Linear(300, action_dim),  # output layer
            nn.Tanh()
        )
        self.action_bound = action_bound

    def forward(self, x):
        return self.net(x) * self.action_bound  # output action scaled

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),  # input: state + action
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)  # output: Q-value
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))  # concat state and action

#Noise
class OUNoise:
    def __init__(self, size, mu=0.0, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(size) * self.mu  # initialize noise state

    def reset(self):
        self.state = np.ones_like(self.state) * self.mu  # reset noise at start of episode

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))  # OU process formula
        self.state = x + dx
        return self.state

#Initialize 
actor = Actor(state_dim, action_dim, action_bound).to(device)  # main actor
target_actor = Actor(state_dim, action_dim, action_bound).to(device)  # target actor
critic = Critic(state_dim, action_dim).to(device)  # main critic
target_critic = Critic(state_dim, action_dim).to(device)  # target critic

target_actor.load_state_dict(actor.state_dict())  # copy weights initially
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

buffer = ReplayBuffer()
noise = OUNoise(action_dim)

# plot
plt.ion()
fig, ax = plt.subplots()
ep_rewards = []
avg_rewards = []
reward_line, = ax.plot([], [], label='Episode Reward')
avg_line, = ax.plot([], [], label='Moving Avg Reward', color='orange')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('DDPG Humanoid Training')
ax.legend()
ax.grid()

def update_plot():
    reward_line.set_xdata(np.arange(len(ep_rewards)))  # update raw rewards
    reward_line.set_ydata(ep_rewards)
    avg_line.set_xdata(np.arange(len(avg_rewards)))  # update moving avg
    avg_line.set_ydata(avg_rewards)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

best_episode_reward = -np.inf  # initialize very low
best_avg_reward = -np.inf

# Training Loop
for episode in range(EPISODES):
    state, _ = env.reset()
    noise.reset()  # reset noise every episode
    episode_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # convert state

        with torch.no_grad():
            action = actor(state_tensor).cpu().numpy()[0]  # get action from actor

        action = np.clip(action + noise.sample(), -action_bound, action_bound)  # add noise for exploration

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reward shaping 
        alive_bonus = info.get('reward_survive', 0)  # get alive bonus from Mujoco
        forward_bonus = info.get('reward_forward', 0)  # forward progress
        standing_bonus = 2.0 if next_state[2] > 1.0 else 0.0  # extra bonus if torso is high
        forward_velocity_bonus = 2.0 * forward_bonus  # scaled forward bonus
        reward += standing_bonus + forward_velocity_bonus  # combine bonuses
        reward *= REWARD_SCALE  # optional scaling

        buffer.put((state, action, reward, next_state, done))  # save transition
        state = next_state  # move to next state
        episode_reward += reward  # accumulate episode reward

        # Start updating after enough samples
        if buffer.size() > START_TRAIN_AFTER:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                target_actions = target_actor(next_states)  # target actor's action
                target_q = target_critic(next_states, target_actions)  # target Q-value
                y = rewards + GAMMA * target_q * (1 - dones)  # compute target value

            q_val = critic(states, actions)  # critic's current Q-value
            critic_loss = nn.MSELoss()(q_val, y)  # critic loss

            critic_optimizer.zero_grad()
            critic_loss.backward()  # update critic
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()  # actor loss: maximize Q

            actor_optimizer.zero_grad()
            actor_loss.backward()  # update actor
            actor_optimizer.step()

            # soft update for target critic
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            # soft update for target actor
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        if done:
            break

    # update noise sigma
    noise.sigma = max(SIGMA_MIN, noise.sigma * NOISE_DECAY)
    ep_rewards.append(episode_reward)
    avg_rewards.append(np.mean(ep_rewards[-MOVING_AVG_WINDOW:]))

    update_plot()
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Moving Avg: {avg_rewards[-1]:.2f}, Noise sigma: {noise.sigma:.3f}")

    # save model
    if episode % SAVE_INTERVAL == 0 and episode != 0:
        torch.save(actor.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # Check and save best policies
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        torch.save(actor.state_dict(), "best_episode_policy.pth")
        print(f"New best episode reward model saved: {best_episode_reward:.2f}")

    if avg_rewards[-1] > best_avg_reward:
        best_avg_reward = avg_rewards[-1]
        torch.save(actor.state_dict(), "best_avg_policy.pth")
        print(f"New best moving avg reward model saved: {best_avg_reward:.2f}")

plt.ioff()
plt.savefig("ddpg_humanoid_rewards.png")
plt.show()
env.close()
