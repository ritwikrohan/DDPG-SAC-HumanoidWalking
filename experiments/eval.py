import gymnasium as gym
import torch
import numpy as np

# ----------- Load environment -----------
ENV_NAME = "Humanoid-v5"
env = gym.make(ENV_NAME, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ----------- Define Actor (same as training) -----------
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(256, action_dim)
        self.log_std = torch.nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state, deterministic=True):
        mu, std = self.forward(state)
        if deterministic:
            action = torch.tanh(mu) * self.max_action
            return action
        else:
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            return y_t * self.max_action

# ----------- Create actor and load weights -----------
actor = Actor(state_dim, action_dim, max_action)
checkpoint = torch.load("sac2_checkpoint_1000000.pt", map_location=torch.device("cpu"))  # <-- Change file if needed
actor.load_state_dict(checkpoint["actor"])
actor.eval()

# ----------- Evaluation loop -----------
NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = actor.sample(state_tensor, deterministic=True)
        action = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state

    print(f"Episode {ep}, Reward: {episode_reward:.2f}")

env.close()
