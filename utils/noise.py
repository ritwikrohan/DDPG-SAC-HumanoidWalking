import numpy as np 
from config import NOISE_THETA, NOISE_SIGMA

class OUNoise:
    def __init__(self, size, mu=0.0, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        # OU noise helps create smooth, continuous noise for actions (better than just random jumps)
        # Useful for continuous control tasks like moving humanoid joints smoothly
        self.mu = mu              # The mean value the noise will move towards (usually 0)
        self.theta = theta        # Controls how strongly noise is pulled back towards the mean (like a "spring force")
        self.sigma = sigma        # Standard deviation, controls how big the random movements are (how noisy it is)
        self.state = np.ones(size) * self.mu  # Start with all values at mu (0). 'size' = number of actions (17 for Humanoid)

    def reset(self):
        # Reset noise state back to mean (0) at the start of each episode
        self.state = np.ones_like(self.state) * self.mu

    def sample(self):
        # Compute change in noise (dx)
        # dx = theta * (mu - current state) + sigma * random normal value
        # First term pulls state back to mean (like a rubber band), second term adds random kick
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx  # Update noise state by adding dx
        return self.state  # Return current noise to add to action
