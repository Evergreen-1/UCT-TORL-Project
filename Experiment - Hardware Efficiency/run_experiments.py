import argparse
import subprocess
import sys

SCRIPTS = {
    "DT": "dt.py",
    "Linear-DT": "linear-dt.py",
    "SOFT-DT": "soft-dt.py",
}

ENV_CONFIGS = {
    "Walker2d": {
        "env_name": "Walker2d-v5",
        "dataset_id": "mujoco/walker2d/medium-v0",
        "target_returns": "[6200.0, 3100.0]",
    },
    "HalfCheetah": {
        "env_name": "HalfCheetah-v5",
        "dataset_id": "mujoco/halfcheetah/medium-v0",
        "target_returns": "[14200.0, 7100.0]",
    },
}

SEEDS = [0, 1, 2, 3, 4]
SEQ_LENS = [20, 40, 60, 80]

def parse_args():
    parser = argparse.ArgumentParser(description="Run the decision transformer experiments for Walker2d or HalfCheetah.")
    parser.add_argument(
        "--env",
        choices=["Walker2d", "HalfCheetah"],
        default="Walker2d",
        help="Choose the environment configuration to use for all scripts.",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=SEQ_LENS,
        help="Sequence lengths to test. Defaults to 20 40 60 80.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds to test. Defaults to 0 1 2 3 4.",
    )
    return parser.parse_args()

def run_experiment(transformer_type: str, seed: int, seq_len: int, env_key: str):
    script = SCRIPTS.get(transformer_type)
    env_config = ENV_CONFIGS[env_key]

    cmd = [
        sys.executable,
        script,
        f"--group={transformer_type}-{env_key}",
        f"--train_seed={seed}",
        f"--seq_len={seq_len}",
        f"--env_name={env_config['env_name']}",
        f"--dataset_id={env_config['dataset_id']}",
        f"--target_returns={env_config['target_returns']}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()

    for transformer_type in ("DT", "Linear-DT", "SOFT-DT"):
        for seq_len in args.seq_lens:
            for seed in args.seeds:
                run_experiment(transformer_type, seed, seq_len, args.env)

if __name__ == "__main__":
    main()