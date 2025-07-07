import gymnasium as gym

# creating mujoco environment as per documentation
env = gym.make("Humanoid-v5", render_mode="human")

obs, info = env.reset()

# running some sample actions
for _ in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action) # take one action
    # if humanoid fails
    if terminated or truncated:
        obs, info = env.reset()

# not used any RL algorithm as of now, Above opny spawns the environment and does some random actions
env.close()
