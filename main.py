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

# Create the Humanoid environment (render_mode so I can see it in action)
env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]  # Number of state features (~376 for Humanoid)
action_dim = env.action_space.shape[0]      # Number of actions (17 torques for Humanoid)
action_bound = env.action_space.high[0]     # Max absolute value for action (torque limit)

# Create Actor network and copy it to target
actor = Actor(state_dim, action_dim, action_bound).to(device)            # Main actor network
target_actor = Actor(state_dim, action_dim, action_bound).to(device)     # Target actor (slow updates)

# Create Critic network and copy it to target
critic = Critic(state_dim, action_dim).to(device)            # Main critic network
target_critic = Critic(state_dim, action_dim).to(device)     # Target critic

# Initialize target networks with same weights as original networks
target_actor.load_state_dict(actor.state_dict())  
target_critic.load_state_dict(critic.state_dict())

# Create optimizers for Actor and Critic
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)    # Optimizer for actor
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR) # Optimizer for critic

# Initialize replay buffer and noise generator
buffer = ReplayBuffer()           # Memory to store past experiences
noise = OUNoise(action_dim)       # OU noise to encourage exploration
plotter = Plotter()              # For live plotting of rewards

# Create folders if they don't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

best_avg_reward = -np.inf  # Start with very low best reward

# Training Loop
for episode in range(EPISODES):  # Loop over episodes
    state, _ = env.reset()        # Reset environment to get starting state
    noise.reset()                # Reset noise state for smooth fresh start
    episode_reward = 0          # Total reward counter for this episode

    for step in range(MAX_STEPS):  # Loop over steps within this episode
        # Convert state to tensor for actor input
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  
        with torch.no_grad():   # Do not track gradients while choosing action
            action = actor(state_tensor).cpu().numpy()[0]  # Get action from actor, convert to numpy array
        # Add noise to make action exploratory
        action = np.clip(action + noise.sample(), -action_bound, action_bound)  # Clip to valid range

        # Perform the action in environment
        next_state, reward, terminated, truncated, info = env.step(action)  
        done = terminated or truncated  # Check if episode is over

        # Reward shaping
        alive_bonus = info.get('reward_survive', 0)            # Bonus for staying alive
        forward_bonus = info.get('reward_forward', 0)          # Bonus for moving forward
        standing_bonus = 2.0 if next_state[2] > 1.0 else 0.0  # Extra bonus if torso stays high (standing)
        forward_velocity_bonus = 2.0 * forward_bonus           # Extra boost for higher forward velocity
        reward += standing_bonus + forward_velocity_bonus     # Add bonuses to original reward
        reward *= REWARD_SCALE                                # Scale final reward

        # Store experience in replay buffer
        buffer.put((state, action, reward, next_state, done))
        state = next_state            # Move to next state
        episode_reward += reward    # Add reward to this episode's total

        # Only start learning once we have enough experiences
        if buffer.size() > START_TRAIN_AFTER:
            # Sample random batch
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # Compute target Q-values using target networks (critic and actor)
            with torch.no_grad():
                target_actions = target_actor(next_states)
                target_q = target_critic(next_states, target_actions)
                y = rewards + GAMMA * target_q * (1 - dones)  # Bellman update

            # Critic loss: MSE between predicted Q and target Q
            q_val = critic(states, actions)
            critic_loss = torch.nn.MSELoss()(q_val, y)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor loss: maximize Q value predicted by critic (negative sign because we minimize loss)
            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target critic parameters
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            # Soft update target actor parameters
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        if done:  # Stop loop if episode is over
            break

    # Decay noise after each episode so exploration gradually reduces
    noise.sigma = max(SIGMA_MIN, noise.sigma * NOISE_DECAY)

    # Calculate moving average reward
    avg_reward = np.mean(plotter.ep_rewards[-MOVING_AVG_WINDOW:] if plotter.ep_rewards else [episode_reward])
    plotter.update(episode_reward, avg_reward)  # Update plot with new data

    # Save best model if current avg_reward is higher
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(actor.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved with moving avg reward: {best_avg_reward:.2f}")

    # Print info for this episode
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Moving Avg: {avg_reward:.2f}, Noise sigma: {noise.sigma:.3f}")

    # Periodically save checkpoint
    if episode % SAVE_INTERVAL == 0 and episode != 0:
        torch.save(actor.state_dict(), MODEL_PATH)
        print(f"Model checkpoint saved to {MODEL_PATH}")

# After all episodes, save final plot and close environment
plotter.save()
env.close()