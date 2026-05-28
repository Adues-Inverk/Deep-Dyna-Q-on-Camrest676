"""
Train k=0 (pure DQN) and k=5 (Dyna-Q) on the Frames hotel-booking dataset
with identical hyperparameters so the curves can be compared cleanly.

Usage:
    python train_frames_comparison.py          # trains both sequentially
    python train_frames_comparison.py --k 0    # only k=0
    python train_frames_comparison.py --k 5    # only k=5
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def make_cmd(k: int, output_dir: str, episodes: int, seed: int) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    return [
        sys.executable, os.path.join(ROOT, "run_frames.py"),
        "--agt",                     "9",
        "--episodes",                str(episodes),
        "--max_turn",                "20",
        "--dqn_hidden_size",         "256",
        "--batch_size",              "64",
        "--gamma",                   "0.99",
        "--learning_rate",           "1e-3",
        "--target_tau",              "0.005",
        "--epsilon",                 "0.5",
        "--min_epsilon",             "0.05",
        "--epsilon_decay",           "0.992",
        "--experience_replay_pool_size", "10000",
        "--per_alpha",               "0.6",
        "--per_beta",                "0.4",
        "--world_model_weight",      "0.5" if k > 0 else "0.0",
        "--planning_steps",          str(k),
        "--train_world_model",       "1" if k > 0 else "0",
        "--boosted",                 "1",
        "--warm_start",              "1",
        "--warm_start_epochs",       "100",
        "--simulation_epoch_size",   "100",
        "--eval_baseline_episodes",  "100",
        "--success_rate_threshold",  "0.4",
        "--write_model_dir",         output_dir,
        "--save_check_point",        "25",
        "--seed",                    str(seed),
        "--torch_seed",              "100",
        "--sample_dialog_seed",      "424242",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=None, help="Only train this k (0 or 5).")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    base = os.path.join(ROOT, "deep_dialog", "checkpoints", "frames_best")
    configs = [
        {"k": 0, "label": "k=0 (pure DQN baseline)",
         "dir": os.path.join(base, "k0")},
        {"k": 5, "label": "k=5 (Dyna-Q — world-model planning)",
         "dir": os.path.join(base, "k5")},
    ]

    if args.k is not None:
        configs = [c for c in configs if c["k"] == args.k]
        if not configs:
            sys.exit(f"Unknown k={args.k}; choose 0 or 5.")

    for cfg in configs:
        sep = "=" * 65
        print(f"\n{sep}\nTraining: {cfg['label']}\nOutput:   {cfg['dir']}\n{sep}\n")
        cmd = make_cmd(cfg["k"], cfg["dir"], args.episodes, args.seed)
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            print(f"\nWARNING: k={cfg['k']} run exited with code {ret.returncode}")
        else:
            print(f"\nDone: {cfg['label']}")


if __name__ == "__main__":
    main()
