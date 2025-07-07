# DDPG Humanoid Walker

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to learn humanoid walking behavior using Gymnasium's `Humanoid-v5` environment (MuJoCo-based).

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
git clone https://github.com/ritwikrohan/humanoid_walking_ddpg.git
cd humanoid-ddpg

python3 -m venv venv
source venv/bin/activate

# Install PyTorch with correct CUDA version (example for CUDA 12.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install other dependencies (optional)
pip install -r requirements.txt

python3 main.py
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