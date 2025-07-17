# Humanoid Walker

This project implements a **Soft Actor-Critic (SAC)** agent to learn humanoid walking behavior using Gymnasium's `Humanoid-v5` environment (MuJoCo-based).  
Earlier attempts with DDPG were less stable; SAC provided smoother and more robust walking performance.

---

## Algorithms

-  **SAC** (main)
-  DDPG (experimental, included for comparison)

---

## Project Structure

- `main.py` — Entry point to run training.
- `eval.py` — Script to evaluate a saved policy without noise.
- `config.py` — Hyperparameters and paths.
- `utils/` — Supporting modules (networks, buffer, noise, plotter).
- `checkpoints/` — Saved model checkpoints.
- `results/` — Reward plots.
- `requirements.txt` — Python dependencies.

## Setup and Installation

```bash
git clone https://github.com/ritwikrohan/humanoid_walking.git
cd humanoid_walking_ddpg

python3 -m venv venv
source venv/bin/activate

# Install PyTorch with correct CUDA version (example for CUDA 12.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install other dependencies
pip install -r requirements.txt

python3 main_sac.py
```
- Trains the humanoid walker from scratch.
- Saves the best model to `checkpoints/ddpg_humanoid_best.pth`
- Generates reward plots in `results/`

## Evaluation

```bash
python3 eval.py
```

- Loads the best saved actor model.
- Runs episodes without noise to check pure policy performance.ddpg_humanoid_best.pth`
- Prints total reward per episode and average reward across runs.
- Opens MuJoCo viewer to visualize humanoid walking.

## Demo

Below is a sample result of the trained SAC agent walking in the MuJoCo `Humanoid-v5` environment.

![Humanoid Walking Demo](media/humanoid_walking.gif)
