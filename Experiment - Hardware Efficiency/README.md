# Transformers for Offline RL - Lightweight Variants of the DT

This project contains three training scripts:

- `dt.py`
- `linear-dt.py`
- `soft-dt.py`

They train Decision Transformer variants on either Walker2d or HalfCheetah environments using Minari datasets and Weights & Biases logging.

This project was adapted from the CORL repository: https://github.com/corl-team/CORL

For more information, see the paper: https://openreview.net/forum?id=SyAS49bBcv

## Project structure

- `dt.py` — standard Decision Transformer (DT)
- `linear-dt.py` — linear-attention variant
- `soft-dt.py` — SOFT-attention variant
- `run_experiments.py` — runs the full sweep across seeds and sequence lengths
- `sweep_linear.yaml` — W&B Bayesian hyperparameter sweep configuration for `linear-dt.py`
- `DatasetVerify.py` — checks the Minari dataset distribution and helps choose accurate returns-to-go (RTG) values before training
- `requirements/requirements.txt` — runtime dependencies
- `requirements/requirements_dev.txt` — dev dependencies

## Requirements

Python 3.10 or 3.11 is recommended.

Install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\requirements.txt
```

For developer tools:

```powershell
python -m pip install -r requirements\requirements_dev.txt
```

> **Note:** The default `torch` install pulls a CUDA 12.4 build (`+cu124`). This isn't a strict
> requirement — it just needs to be compatible with your installed NVIDIA driver, which doesn't
> need to exactly match 12.4. Check your driver's supported CUDA version with `nvidia-smi` before
> installing.

### Installing pytorch-fast-transformers

After installing the requirements, install pytorch-fast-transformers
without build isolation:

python -m pip install pytorch-fast-transformers==0.4.0 --no-build-isolation

## Quick run

Run a single training job directly:

```powershell
python dt.py --train_seed=0 --seq_len=20 --env_name=Walker2d-v5 --dataset_id=mujoco/walker2d/medium-v0 --target_returns="[6200.0, 3100.0]"
```

Run the linear model:

```powershell
python linear-dt.py --train_seed=0 --seq_len=20 --env_name=Walker2d-v5 --dataset_id=mujoco/walker2d/medium-v0 --target_returns="[6200.0, 3100.0]"
```

Run the SOFT model:

```powershell
python soft-dt.py --train_seed=0 --seq_len=20 --env_name=Walker2d-v5 --dataset_id=mujoco/walker2d/medium-v0 --target_returns="[6200.0, 3100.0]"
```

## Run the full experiment

The helper script runs all three models across several sequence lengths and seeds.

For Walker2d:

```powershell
python run_experiments.py --env Walker2d
```

For HalfCheetah:

```powershell
python run_experiments.py --env HalfCheetah
```

Optional custom values:

```powershell
python run_experiments.py --env Walker2d --seq-lens 20 40 --seeds 0 1
```

## Run the W&B Linear-DT sweep

The `sweep_linear.yaml` configuration defines a Bayesian W&B sweep for `linear-dt.py` on the Walker2d medium dataset. It runs up to 25 jobs and maximizes the normalized evaluation score. The sweep searches over learning rate, model architecture, number of layers, sequence length, attention dropout, and residual dropout while keeping the main training settings fixed.

Create the sweep and start an agent with:

```powershell
wandb sweep sweep_linear.yaml
wandb agent TORL-team/Experiment-B/<sweep_id>
```

The configuration uses CUDA, 30,000 update steps, evaluation every 5,000 steps, gradient clipping at `0.25`, weight decay of `1e-4`, and stores checkpoints in `./sweep_checkpoints`. Replace `<sweep_id>` with the ID printed by `wandb sweep`.

## Environment values used by the helper

The script maps the environment name to the corresponding config values:

### Walker2d

- `env_name`: `Walker2d-v5`
- `dataset_id`: `mujoco/walker2d/medium-v0`
- `target_returns`: `[6200.0, 3100.0]`

### HalfCheetah

- `env_name`: `HalfCheetah-v5`
- `dataset_id`: `mujoco/halfcheetah/medium-v0`
- `target_returns`: `[14200.0, 7100.0]`

## GPU vs CPU

The scripts default to CUDA when available, and automatically fall back to CPU if CUDA is not available in the `__post_init__` logic in the training config for the scripts. If you want to force CPU, pass `--device=cpu` if supported by your local command line, or set the `device` field in the config to `cpu`.

## W&B logging

The project uses Weights & Biases for experiment tracking. If W&B is not configured, you may need to log in before running:

```powershell
wandb login
```

## Output files

The scripts may save:

- checkpoints under a checkpoint path if configured
- evaluation videos in a folder
- run metrics to the W&B dashboard

## Troubleshooting

If imports fail:

- make sure the environment is activated
- reinstall the requirements from [requirements/requirements.txt](requirements/requirements.txt)

If training fails because of a dataset or environment issue:

- confirm the environment name is correct
- confirm the dataset ID matches the environment
- ensure the environment dependencies are installed

If the GPU is unavailable:

- run on CPU for a smoke test
- use a CUDA-capable machine or cloud GPU for full experiments

## Notes

This project is intended for training RL experiments and may take a long time to complete on a typical workstation. The full sweep in `run_experiments.py` runs many seeds and sequence lengths, so it is best suited for a machine with a GPU.

All the runs and sweeps for the experiments can be found at: https://wandb.ai/TORL-team/Experiment-B-Final