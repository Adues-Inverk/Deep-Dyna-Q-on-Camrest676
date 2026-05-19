"""
Sweep DDQ planning_steps K in {1, 3, 5, 7, 10} on CamRest676 with all
other hyperparameters held constant. Each run goes to its own folder:

    deep_dialog/checkpoints/sweep_k/k_1/
    deep_dialog/checkpoints/sweep_k/k_3/
    ...

Per-run console output is teed to:
    deep_dialog/checkpoints/sweep_k/k_K/train.log

Run:
    python sweep_k.py
    python sweep_k.py --episodes 100        # longer training
    python sweep_k.py --ks 1,5,10           # subset
"""

import argparse
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ROOT_OUT = os.path.join(HERE, "deep_dialog", "checkpoints", "sweep_k")


def run_one(k, episodes, warm_start_epochs, max_turn, seed):
    out_dir = os.path.join(ROOT_OUT, "k_%d" % k)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    print("\n" + "=" * 60)
    print("[K=%d] writing to %s" % (k, out_dir))
    print("=" * 60)
    cmd = [
        PY, os.path.join(HERE, "run_camrest.py"),
        "--agt", "9",
        "--episodes", str(episodes),
        "--warm_start", "1",
        "--warm_start_epochs", str(warm_start_epochs),
        "--simulation_epoch_size", "50",
        "--planning_steps", str(k),
        "--max_turn", str(max_turn),
        "--seed", str(seed),
        "--write_model_dir", out_dir + os.sep,
    ]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print("[K=%d] done in %.1fs, return code %d, log %s"
          % (k, dt, proc.returncode, log_path))
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,3,5,7,10",
                    help="comma-separated K values")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--warm_start_epochs", type=int, default=100)
    ap.add_argument("--max_turn", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5,
                    help="shared seed across all K runs for fair comparison")
    args = ap.parse_args()
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    os.makedirs(ROOT_OUT, exist_ok=True)
    print("PY =", PY)
    print("K values =", ks)
    print("episodes per K =", args.episodes)
    print("warm_start_epochs =", args.warm_start_epochs)
    print("max_turn =", args.max_turn)
    print("seed =", args.seed)

    t_overall = time.time()
    rcs = {}
    for k in ks:
        rcs[k] = run_one(k, args.episodes, args.warm_start_epochs,
                         args.max_turn, args.seed)
    print("\noverall elapsed: %.1fs" % (time.time() - t_overall))
    print("return codes:", rcs)


if __name__ == "__main__":
    main()
