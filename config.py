ENV_NAME = "Humanoid-v5"  # Name of the Gymnasium environment we are using in Mujoco (Humanoid robot)

ACTOR_LR = 1e-4           # Learning rate for the actor network (controls how fast it learns)
CRITIC_LR = 1e-3          # Learning rate for the critic network (usually higher than actor)

GAMMA = 0.995             # Discount factor for future rewards (close to 1 means we care about long-term rewards)
TAU = 0.001               # Soft update rate for target networks (small value makes updates slow and smooth)

BATCH_SIZE = 256         # Number of samples used to update the networks at once (mini-batch size)
MEMORY_SIZE = 1_000_000  # Maximum size of the replay buffer (stores past experiences)

EPISODES = 20005         # Total number of episodes to train
MAX_STEPS = 1000         # Max steps per episode before terminating

START_TRAIN_AFTER = 5000 # Start training only after collecting these many samples (lets buffer fill first)
SAVE_INTERVAL = 500      # How often (episodes) we save a checkpoint model

NOISE_SIGMA = 0.3        # Initial sigma (standard deviation) for exploration noise (Ornstein-Uhlenbeck noise)
NOISE_THETA = 0.15       # Theta parameter for OU noise (controls the "pull" toward mean)
NOISE_DECAY = 0.995      # How fast the noise sigma decays after each episode (reduces exploration over time)
SIGMA_MIN = 0.05         # Minimum sigma value to prevent noise from disappearing completely

REWARD_SCALE = 1.0       # Multiplier to scale the shaped rewards (can tune to make rewards larger or smaller)

MODEL_PATH = "checkpoints/ddpg_humanoid_checkpoint.pth"  # File path for periodic checkpoints
BEST_MODEL_PATH = "checkpoints/ddpg_humanoid_best_till_now.pth"   # File path for saving the best performing model

MOVING_AVG_WINDOW = 50   # Window size to calculate moving average reward (used for smoother plots)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU if available, else CPU