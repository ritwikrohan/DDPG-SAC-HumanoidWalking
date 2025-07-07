import numpy as np
from config import NOISE_THETA, NOISE_SIGMA

class OUNoise:
    def __init__(self, size, mu=0.0, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(size) * self.mu

    def reset(self):
        self.state = np.ones_like(self.state) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state
