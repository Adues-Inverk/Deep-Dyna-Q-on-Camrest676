"""
Focused comparison: K=0 (non-DDQ baseline) vs K=1,3,5 (light planning).
Highlights how even minimal planning stabilizes the policy.

Output files in slides_figures/:
    25_compare_K0_vs_K135_success_rate.png
    25_compare_K0_vs_K135_avg_turns.png
    25_compare_K0_vs_K135_avg_reward.png
    25_compare_K0_vs_K135_summary_table.png

Run:
    python plotting/compare_k_focused.py
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Add project root to sys.path for deep_dialog imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SWEEP_ROOT = os.path.join(PROJECT_ROOT, "deep_dialog", "checkpoints", "sweep_k")
OUT = os.path.join(PROJECT_ROOT, "slides_figures")
os.makedirs(OUT, exist_ok=True)

FOCUS_KS = [0, 1, 3, 5]  # K values to compare
COLORS_MAP = {0: "#c0392b", 1: "#e67e22", 3: "#f39c12", 5: "#27ae60"}
LABELS_MAP = {0: "K=0 (no planning)", 1: "K=1", 3: "K=3", 5: "K=5"}


def load_one(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = sorted(int(k) for k in data["success_rate"].keys() if int(k) >= 0)
    x = keys
    sr = [data["success_rate"][str(k)] for k in keys]
    at = [data["ave_turns"][str(k)] for k in keys]
    ar = [data["ave_reward"][str(k)] for k in keys]
    return {"x": x, "success_rate": sr, "ave_turns": at, "ave_reward": ar, "data": data}


def discover_runs():
    """Map K -> path to performance_records.json under sweep_k/."""
    runs = {}
    for kdir in sorted(glob.glob(os.path.join(SWEEP_ROOT, "k_*"))):
        m = re.match(r"k_(\d+)$", os.path.basename(kdir))
        if not m:
            continue
        k = int(m.group(1))
        pr = os.path.join(kdir, "agt_9_performance_records.json")
        if os.path.isfile(pr):
            runs[k] = pr
    return runs


def smooth(y, w=5):
    """Simple moving average; returns same-length array."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = max(1, min(w, len(y)))
    out = np.convolve(y, np.ones(w) / w, mode="same")
    return out


def plot_metric_focused(runs_data, metric_key, title, ylabel, fname,
                        smooth_window=5, ymin=None, ymax=None, hline=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for k in FOCUS_KS:
        if k not in runs_data:
            continue
        rec = runs_data[k]
        x = rec["x"]
        y = rec[metric_key]
        color = COLORS_MAP.get(k, "#999")
        label = LABELS_MAP.get(k, f"K={k}")

        # Plot raw curve faintly
        ax.plot(x, y, color=color, lw=0.9, alpha=0.25)
        # Plot smoothed curve prominently
        ax.plot(x, smooth(y, smooth_window), color=color, lw=2.5, label=label)

    ax.set_xlabel("Training episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if hline is not None:
        ax.axhline(hline, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def build_summary(runs_data):
    """Per-K aggregate stats for focus set."""
    rows = []
    for k in FOCUS_KS:
        if k not in runs_data:
            continue
        rec = runs_data[k]
        sr = rec["success_rate"]
        at = rec["ave_turns"]
        ar = rec["ave_reward"]
        if not sr:
            continue
        best_idx = int(np.argmax(sr))
        last20 = sr[-20:] if len(sr) >= 20 else sr
        last20_at = at[-20:] if len(at) >= 20 else at
        last20_ar = ar[-20:] if len(ar) >= 20 else ar
        rows.append({
            "K": k,
            "peak_sr": max(sr),
            "peak_ep": rec["x"][best_idx],
            "final_sr_mean20": float(np.mean(last20)),
            "final_turns_mean20": float(np.mean(last20_at)),
            "final_reward_mean20": float(np.mean(last20_ar)),
            "episodes_total": len(sr),
        })
    return rows


def plot_summary_table(rows):
    fig, ax = plt.subplots(figsize=(9.5, 0.6 + 0.45 * (len(rows) + 1)))
    ax.axis("off")
    headers = ["K", "Peak SR", "Peak ep",
               "Final SR (m20)", "Final turns (m20)", "Final reward (m20)"]
    cells = []
    for r in rows:
        cells.append([
            "0 (no plan)" if r["K"] == 0 else "%d" % r["K"],
            "%.3f" % r["peak_sr"],
            "%d" % r["peak_ep"],
            "%.3f" % r["final_sr_mean20"],
            "%.1f" % r["final_turns_mean20"],
            "%.2f" % r["final_reward_mean20"],
        ])
    tbl = ax.table(cellText=cells, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#dfe6f5")
        tbl[(0, j)].set_text_props(weight="bold")
    # Highlight K=0 row
    tbl[(1, 0)].set_facecolor("#ffe6e6")
    ax.set_title("Non-DDQ vs light planning comparison",
                 fontsize=12, pad=8)
    p = os.path.join(OUT, "25_compare_K0_vs_K135_summary_table.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def main():
    runs = discover_runs()
    missing = [k for k in FOCUS_KS if k not in runs]
    if missing:
        print(f"Warning: missing K values {missing}")
    if not any(k in runs for k in FOCUS_KS):
        print("No K-sweep results found.")
        return

    print("Discovered runs (focus set):")
    for k in FOCUS_KS:
        if k in runs:
            print(f"  K={k}  {runs[k]}")

    runs_data = {k: load_one(runs[k]) for k in FOCUS_KS if k in runs}

    plot_metric_focused(runs_data, "success_rate",
                        "Success rate: non-DDQ (K=0) vs light planning (K=1,3,5)",
                        "Success rate", "25_compare_K0_vs_K135_success_rate.png",
                        ymin=-0.02, ymax=1.02)
    plot_metric_focused(runs_data, "ave_turns",
                        "Dialogue length: does planning reduce turns?",
                        "Average turns / dialogue",
                        "25_compare_K0_vs_K135_avg_turns.png")
    plot_metric_focused(runs_data, "ave_reward",
                        "Reward: stability gain from planning",
                        "Average reward",
                        "25_compare_K0_vs_K135_avg_reward.png",
                        hline=0.0)

    rows = build_summary(runs_data)
    plot_summary_table(rows)


if __name__ == "__main__":
    main()
