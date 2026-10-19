# Tabular Q-learning on CliffWalking 

A tabular Q-learning agent for Gymnasium's `CliffWalking-v1` 
(4×12 grid, 48 states, 4 actions). The script trains a baseline, sweeps α, 
γ and ε-decay, selects a best configuration, records a video, and 
runs an action-noise experiment.

## Files

| Path | Description |
|---|---|
| `cliffwalking.py` | All code: Q-learning, evaluation, experiments, plots, video |
| `requirements.txt` | Python dependencies |
| `results/` | Plots: baseline, α / γ / ε-decay sweeps, noise curves |
| `videos/` | MP4 of the best agent following its greedy policy |

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Run
```bash
python cliffwalking.py                  # prints result tables to the terminal
python cliffwalking.py > results/output.txt     # or save the printed output to a file
```

Plots go to `results/` and
the video to `videos/`. Plot y-axes use a symmetric-log scale.

## Settings and results

| | α | γ | ε-decay | Eval reward | Success | Steps | Train reward (last 100) | Episodes to reach -30 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.1 | 0.99 | 0.99 | -13.0 | 1.00 | 13.0 | -49.3 | 259 |
| Best | 0.9 | 0.9 | 0.95 | -13.0 | 1.00 | 13.0 | -46.8 | 155 |

Both use ε from 1.0 to 0.1 over 500 episodes and learn the optimal 13-step
path. Results are averaged over 10 seeds with greedy (ε = 0) evaluation.