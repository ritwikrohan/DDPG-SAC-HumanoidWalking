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
CRITIC_LR = 1e-4
GAMMA = 0.995
TAU = 0.0005
BATCH_SIZE = 512
MEMORY_SIZE = 1_000_000
EPISODES = 20000
MAX_STEPS = 1000
START_TRAIN_AFTER = 10000  # Increased warmup
SAVE_INTERVAL = 500
NOISE_SIGMA = 0.4
NOISE_THETA = 0.15
NOISE_DECAY = 0.999
SIGMA_MIN = 0.2
REWARD_SCALE = 1.0
MODEL_PATH = "ddpg_humanoid_final_tuesday_new_trial.pth"
MOVING_AVG_WINDOW = 50

env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.action_bound = action_bound

    def forward(self, x):
        return self.net(x) * self.action_bound

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class OUNoise:
    def __init__(self, size, mu=0.0, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(size) * self.mu

    def reset(self):
        self.state = np.ones_like(self.state) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

actor = Actor(state_dim, action_dim, action_bound).to(device)
target_actor = Actor(state_dim, action_dim, action_bound).to(device)
critic = Critic(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

buffer = ReplayBuffer()
noise = OUNoise(action_dim)

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
    reward_line.set_xdata(np.arange(len(ep_rewards)))
    reward_line.set_ydata(ep_rewards)
    avg_line.set_xdata(np.arange(len(avg_rewards)))
    avg_line.set_ydata(avg_rewards)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

best_episode_reward = -np.inf
best_avg_reward = -np.inf

for episode in range(EPISODES):
    state, _ = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            if episode < 500:  # extra exploration phase
                action = env.action_space.sample()
            else:
                action = actor(state_tensor).cpu().numpy()[0]
                action = np.clip(action + noise.sample(), -action_bound, action_bound)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reward shaping
        alive_bonus = info.get('reward_survive', 0)
        forward_bonus = info.get('reward_forward', 0)
        standing_bonus = 2.0 if next_state[2] > 1.0 else 0.0
        forward_velocity_bonus = 4.0 * forward_bonus
        fall_penalty = -10.0 if terminated else 0.0
        action_penalty = -0.1 * np.sum(np.square(action))

        reward += standing_bonus + forward_velocity_bonus + fall_penalty + action_penalty
        reward *= REWARD_SCALE

        buffer.put((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        if buffer.size() > START_TRAIN_AFTER:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            with torch.no_grad():
                target_actions = target_actor(next_states)
                target_q = target_critic(next_states, target_actions)
                y = rewards + GAMMA * target_q * (1 - dones)

            q_val = critic(states, actions)
            critic_loss = nn.MSELoss()(q_val, y)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            with torch.no_grad():
                q_val_mean = q_val.mean().item()
                q_val_std = q_val.std().item()
                target_q_mean = target_q.mean().item()
                target_q_std = target_q.std().item()
            print(f"Critic Q mean: {q_val_mean:.2f}, std: {q_val_std:.2f} | Target Q mean: {target_q_mean:.2f}, std: {target_q_std:.2f}")

        if done:
            break

    noise.sigma = max(SIGMA_MIN, noise.sigma * NOISE_DECAY)
    ep_rewards.append(episode_reward)
    avg_rewards.append(np.mean(ep_rewards[-MOVING_AVG_WINDOW:]))

    update_plot()
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Moving Avg: {avg_rewards[-1]:.2f}, Noise sigma: {noise.sigma:.3f}")

    if episode % SAVE_INTERVAL == 0 and episode != 0:
        torch.save(actor.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        torch.save(actor.state_dict(), "best_episode_policy_tuesday.pth")
        print(f"New best episode reward model saved: {best_episode_reward:.2f}")

    if avg_rewards[-1] > best_avg_reward:
        best_avg_reward = avg_rewards[-1]
        torch.save(actor.state_dict(), "best_avg_policy_tuesday.pth")
        print(f"New best moving avg reward model saved: {best_avg_reward:.2f}")

plt.ioff()
plt.savefig("ddpg_humanoid_rewards.png")
plt.show()
env.close()
