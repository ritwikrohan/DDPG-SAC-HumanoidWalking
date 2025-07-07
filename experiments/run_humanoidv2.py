import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib.pyplot as plt

# Set device: use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create environment
env = gym.make("Humanoid-v5", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_discrete_actions = 10  # I reduce continuous actions into 10 fixed choices
action_samples = [env.action_space.sample() for _ in range(n_discrete_actions)]

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Initialize networks
q_net = QNetwork(obs_dim, n_discrete_actions).to(device)
target_net = QNetwork(obs_dim, n_discrete_actions).to(device)
target_net.load_state_dict(q_net.state_dict())

# Try to load model if exists
model_path = "dqn_humanoid.pth"
if os.path.exists(model_path):
    q_net.load_state_dict(torch.load(model_path))
    target_net.load_state_dict(torch.load(model_path))
    print("Loaded model from:", model_path)

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
memory = deque(maxlen=100000)

# Hyperparameters
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
update_target_every = 10
num_episodes = 500

# Reward tracker for visualization
episode_rewards = []

# Epsilon-greedy action selection
def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, n_discrete_actions - 1)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = q_net(state)
            return torch.argmax(q_values).item()

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = select_action(obs)
        action = action_samples[action_idx]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((obs, action_idx, reward, next_obs, done))
        obs = next_obs
        total_reward += reward

        # DQN update
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

            obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(device)
            action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
            next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32).to(device)

            q_values = q_net(obs_batch)
            q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values = target_net(next_obs_batch)
                max_next_q_values = next_q_values.max(1)[0]
                target = reward_batch + gamma * max_next_q_values * (1 - done_batch)

            loss = criterion(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % update_target_every == 0:
        target_net.load_state_dict(q_net.state_dict())

    # Save model every 50 episodes
    if episode % 50 == 0:
        torch.save(q_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track and print rewards
    episode_rewards.append(total_reward)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

env.close()

# Plot total rewards over episodes
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Humanoid DQN Performance')
plt.grid(True)
plt.savefig("rewards_plot.png")
plt.show()
