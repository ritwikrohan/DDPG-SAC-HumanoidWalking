import gymnasium as gym  
import torch             
import numpy as np       
from config import *     
from utils.networks import Actor  # need Actor for evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Create the Humanoid environment
env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]  # Number of state features
action_dim = env.action_space.shape[0]      # Number of actions
action_bound = env.action_space.high[0]     # Max action value

# Create actor network and load saved weights
actor = Actor(state_dim, action_dim, action_bound).to(device)
actor.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))  # Load best model path
actor.eval()  # Set actor to evaluation mode (no dropout, no batch norm changes)

NUM_EVAL_EPISODES = 10  # Number of evaluation episodes
total_rewards = []      # Store reward for each episode

for episode in range(NUM_EVAL_EPISODES):
    state, _ = env.reset()  # Reset environment to get initial state
    episode_reward = 0     # Initialize total reward for this episode
    done = False

    for step in range(MAX_STEPS):  # Use same max step limit
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Convert to tensor
        with torch.no_grad():  # No gradients needed
            action = actor(state_tensor).cpu().numpy()[0]  # Get action from actor
        action = np.clip(action, -action_bound, action_bound)  # Clip to valid range

        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Check if episode is finished

        state = next_state  # Move to next state
        episode_reward += reward  # Add reward

        if done:
            break  # Exit loop if episode finished

    total_rewards.append(episode_reward)  # Store total reward for this episode
    print(f"Evaluation Episode {episode+1}: Total Reward = {episode_reward:.2f}")

# After all episodes, calculate average reward
avg_reward = np.mean(total_rewards)
print(f"\nAverage Reward over {NUM_EVAL_EPISODES} Evaluation Episodes: {avg_reward:.2f}")

env.close()  # Close environment after evaluation