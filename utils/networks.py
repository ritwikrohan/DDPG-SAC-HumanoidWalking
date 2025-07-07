import torch 
import torch.nn as nn  

class Actor(nn.Module):  # Actor network: decides which action to take (torques)
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        # My thinking: I use a sequential network to keep it simple and clear
        # First layer: I chose 400 neurons, big enough to capture complex features from high-dimensional state (~376)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),  # Input layer from state vector to 400 hidden units
            nn.ReLU(),                  # ReLU because it's simple and works well in deep networks
            nn.Linear(400, 300),       # Second layer: 400 to 300 neurons, reducing size step by step
            nn.ReLU(),                 # Again ReLU to keep non-linearities
            nn.Linear(300, action_dim), # Output layer: finally get to action dimension (17 torques for Humanoid)
            nn.Tanh()                 # Tanh makes sure outputs are between -1 and 1, then I scale by action_bound
        )
        self.action_bound = action_bound  # save action_bound to multiply later (so torques don't exceed limits)

    def forward(self, x):
        # forward pass returns action vector, scaled by action_bound to match real robot torque range
        return self.net(x) * self.action_bound

class Critic(nn.Module):  # Critic network: scores how good a given action is in a state
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__() 
        # Critic needs both state and action as input because Q-value depends on both
        # concatenate state and action, so input layer size is (state_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),  # First layer combines state and action, then goes to 400 units
            nn.ReLU(),                               # ReLU for non-linearity
            nn.Linear(400, 300),                    # Second hidden layer to reduce size, helps learn compressed features
            nn.ReLU(),                             # Another ReLU to keep non-linearity
            nn.Linear(300, 1)                     # Final output: one single Q-value (score)
        )

    def forward(self, state, action):
        # concatenate state and action before passing to network
        return self.net(torch.cat([state, action], dim=1))  # dim=1 means concatenate columns (features axis)
