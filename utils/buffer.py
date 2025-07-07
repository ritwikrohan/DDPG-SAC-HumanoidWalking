import random  
import torch  
from collections import deque  
from config import MEMORY_SIZE, device  
import numpy as np  

class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)  # Buffer to store (state, action, reward, next_state, done) tuples

    def put(self, transition):
        self.buffer.append(transition)  # Store new experience in buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # Pick random batch_size number of experiences
        states, actions, rewards, next_states, dones = zip(*batch)  # Separate each element into its own list

        # states: current observations (shape: 376 for Humanoid-v5, includes joint positions, velocities, etc.)
        # actions: joint torque values to apply (shape: 17 for Humanoid-v5)
        # rewards: single scalar reward values (float)
        # next_states: observations after taking action
        # dones: 1 if episode finished (fall or terminate), 0 if not

        states = np.array(states)  # Convert list of states to array
        actions = np.array(actions)  # Convert list of actions to array
        rewards = np.array(rewards)  # Convert list of rewards to array
        next_states = np.array(next_states)  # Convert list of next states to array
        dones = np.array(dones)  # Convert list of done flags to array

        return (
            torch.tensor(states, dtype=torch.float32).to(device),  # Convert states to tensor and move to GPU
            torch.tensor(actions, dtype=torch.float32).to(device),  # Convert actions to tensor and move to GPU
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),  # Convert rewards to tensor, shape (batch_size, 1)
            torch.tensor(next_states, dtype=torch.float32).to(device),  # Convert next states to tensor and move to GPU
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)  # Convert dones to tensor, shape (batch_size, 1)
        )

    def size(self):
        return len(self.buffer)  # Return how many experiences we currently have stored
