# DDPG Humanoid Walker

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to learn humanoid walking behavior in Gymnasium's `Humanoid-v5` environment.

## Project Structure

- `main.py` — Entry point to run training.
- `config.py` — Hyperparameters and paths.
- `utils/` — Supporting modules (networks, buffer, noise, plotter).
- `checkpoints/` — Saved model checkpoints.
- `results/` — Reward plots.
- `requirements.txt` — Python dependencies.

## Running

```bash
pip install -r requirements.txt
python3 main.py
