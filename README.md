# Humanoid Walking with Deep Reinforcement Learning

Bipedal locomotion for Humanoid-v5 using SAC and DDPG in MuJoCo physics simulation.

## Overview

Implementation of Soft Actor-Critic (SAC) and Deep Deterministic Policy Gradient (DDPG) algorithms for stable humanoid walking. The project demonstrates learning complex 17-DOF bipedal locomotion in a 376-dimensional state space.

## Demo

Below is a sample result of the trained SAC agent walking in the MuJoCo `Humanoid-v5` environment.

![Humanoid Walking Demo](media/humanoid_walking.gif)

## Key Features

- **Soft Actor-Critic (SAC)**: Primary algorithm with automatic temperature tuning
- **DDPG Baseline**: Comparison implementation with Ornstein-Uhlenbeck noise
- **Reward Shaping**: Standing bonus, forward velocity rewards for stable gait
- **Robust Training**: 8+ hour stable walking without falls after convergence
- **Real-time Visualization**: MuJoCo renderer for monitoring training progress

## Performance Metrics

| Metric | Value | Conditions |
|--------|-------|------------|
| Stable Walking Duration | 8+ hours | Continuous locomotion without falls |
| Learning Speed | 2x faster | SAC vs baseline DDPG |
| Convergence Time | ~20k steps | To achieve stable gait |
| Action Dimensions | 17 DOF | Joint torques |
| State Space | 376-dim | Joint positions, velocities, contacts |
| Average Reward | 6000+ | After convergence |

## Technical Stack

- **Environment**: Gymnasium Humanoid-v5 (MuJoCo)
- **Framework**: PyTorch
- **Algorithms**: SAC with entropy regularization, DDPG
- **Exploration**: Ornstein-Uhlenbeck noise (DDPG), stochastic policy (SAC)
- **Hardware**: CUDA-accelerated training

## Installation

Clone and setup environment:

    git clone https://github.com/ritwikrohan/DDPG-SAC-HumanoidWalking.git
    cd DDPG-SAC-HumanoidWalking
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Install PyTorch with CUDA (adjust for your CUDA version)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    pip install -r requirements.txt

## Usage

Train SAC agent (recommended):

    python3 main_sac.py

Train DDPG agent (baseline):

    python3 main.py

Evaluate trained model:

    python3 eval.py

## Repository Structure

```
DDPG-SAC-HumanoidWalking/
├── main_sac.py              # SAC training loop
├── main.py                  # DDPG training loop  
├── eval.py                  # Evaluation script
├── config.py                # Hyperparameters
├── utils/
│   ├── networks.py         # Actor-Critic networks
│   ├── buffer.py          # Replay buffer
│   ├── noise.py           # OU noise for DDPG
│   └── plotter.py         # Live training plots
├── checkpoints/            # Saved models
└── results/               # Training curves
```

## Technical Implementation

### SAC Algorithm
1. **Stochastic Policy**: Gaussian policy with learnable mean and variance
2. **Entropy Regularization**: Automatic temperature adjustment for exploration
3. **Twin Q-Networks**: Mitigates Q-value overestimation
4. **Soft Updates**: Target network updates with τ=0.005

### DDPG Implementation
1. **Deterministic Policy**: Direct action output with OU noise
2. **Experience Replay**: 1M buffer capacity
3. **Target Networks**: Stabilizes learning with soft updates
4. **Noise Decay**: σ decays from 0.3 to 0.05

### Reward Shaping
- **Survival Bonus**: +1 for each timestep alive
- **Forward Reward**: Proportional to forward velocity
- **Standing Bonus**: +2 when torso height > 1.0m
- **Velocity Scaling**: 2x multiplier on forward movement

## Results

- SAC achieves stable walking after ~20k environment steps
- DDPG shows less stable learning, prone to local optima
- Final policy maintains balance for 8+ hours of continuous walking
- Robust to small perturbations and varying forward speeds

## Contact

**Ritwik Rohan**  
Robotics Engineer | Johns Hopkins MSE '25  
Email: ritwikrohan7@gmail.com  
LinkedIn: [linkedin.com/in/ritwik-rohan](https://linkedin.com/in/ritwik-rohan)

---
