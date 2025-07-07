import gymnasium as gym
import torch
import numpy as np
import os

from config import *
from utils.buffer import ReplayBuffer
from utils.networks import Actor, Critic
from utils.noise import OUNoise
from utils.plotter import Plotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

actor = Actor(state_dim, action_dim, action_bound).to(device)
target_actor = Actor(state_dim, action_dim, action_bound).to(device)
critic = Critic(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

buffer = ReplayBuffer()
noise = OUNoise(action_dim)
plotter = Plotter()

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

best_avg_reward = -np.inf

for episode in range(EPISODES):
    state, _ = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(state_tensor).cpu().numpy()[0]
        action = np.clip(action + noise.sample(), -action_bound, action_bound)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Custom reward shaping
        alive_bonus = info.get('reward_survive', 0)
        forward_bonus = info.get('reward_forward', 0)
        standing_bonus = 2.0 if next_state[2] > 1.0 else 0.0
        forward_velocity_bonus = 2.0 * forward_bonus
        reward += standing_bonus + forward_velocity_bonus
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
            critic_loss = torch.nn.MSELoss()(q_val, y)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        if done:
            break

    noise.sigma = max(SIGMA_MIN, noise.sigma * NOISE_DECAY)

    avg_reward = np.mean(plotter.ep_rewards[-MOVING_AVG_WINDOW:] if plotter.ep_rewards else [episode_reward])
    plotter.update(episode_reward, avg_reward)

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(actor.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved with moving avg reward: {best_avg_reward:.2f}")

    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Moving Avg: {avg_reward:.2f}, Noise sigma: {noise.sigma:.3f}")

    if episode % SAVE_INTERVAL == 0 and episode != 0:
        torch.save(actor.state_dict(), MODEL_PATH)
        print(f"Model checkpoint saved to {MODEL_PATH}")

plotter.save()
env.close()
