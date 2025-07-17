import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
ENV_NAME = "Humanoid-v5"
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
LEARNING_STARTS = 1000
TOTAL_TIMESTEPS = 1_000_000
SAVE_INTERVAL = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("runs2/sac_sb3_style")

class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device))


    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

target_entropy = -action_dim
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

replay_buffer = ReplayBuffer()

global_step = 0
episode = 0

while global_step < TOTAL_TIMESTEPS:
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        global_step += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = actor.sample(state_tensor)
        action = action.cpu().numpy()[0]
        # next_state, reward, terminated, truncated, _ = env.step(action)
        # done = terminated or truncated
        # episode_reward += reward

        # replay_buffer.put((state, action, reward, next_state, float(done)))
        # state = next_state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --------------------
        # Reward shaping
        # --------------------
        alive_bonus = info.get('reward_survive', 0)
        forward_bonus = info.get('reward_forward', 0)
        standing_bonus = 2.0 if next_state[2] > 1.0 else 0.0
        forward_velocity_bonus = 2.0 * forward_bonus

        reward += standing_bonus + forward_velocity_bonus

        REWARD_SCALE = 1.0  # adjust if you want to scale final reward
        reward *= REWARD_SCALE

        # --------------------

        episode_reward += reward

        replay_buffer.put((state, action, reward, next_state, float(done)))
        state = next_state


        if replay_buffer.size() > LEARNING_STARTS:
            for _ in range(1):  # gradient_steps=1
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                new_actions, log_pi = actor.sample(states)
                q1_new, q2_new = critic(states, new_actions)
                q_min = torch.min(q1_new, q2_new)

                # Alpha tuning
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp()

                actor_loss = (alpha * log_pi - q_min).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                with torch.no_grad():
                    next_actions, next_log_pi = actor.sample(next_states)
                    q1_next, q2_next = critic_target(next_states, next_actions)
                    q_target = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = rewards + (1 - dones) * GAMMA * q_target

                q1, q2 = critic(states, actions)
                critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            writer.add_scalar("Loss/actor", actor_loss.item(), global_step)
            writer.add_scalar("Loss/critic", critic_loss.item(), global_step)
            writer.add_scalar("Loss/alpha", alpha_loss.item(), global_step)
            writer.add_scalar("Alpha", alpha.item(), global_step)

        # Save model periodically
        if global_step % SAVE_INTERVAL == 0:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
                "log_alpha": log_alpha,
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }, f"sac2_checkpoint_{global_step}.pt")
            print(f"Checkpoint saved at step {global_step}")

    writer.add_scalar("Reward/episode", episode_reward, global_step)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Step: {global_step}")

    episode += 1

env.close()
writer.close()
