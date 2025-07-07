import random
import torch
from collections import deque
from config import MEMORY_SIZE, device
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of arrays to single NumPy arrays first (fixes warning and speeds up)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device))

    def size(self):
        return len(self.buffer)
