"""
Read every deep_dialog/checkpoints/sweep_k/k_*/agt_9_performance_records.json
and produce overlay comparison plots + a summary table.

Output files in slides_figures/:
    20_compare_K_success_rate.png
    21_compare_K_avg_turns.png
    22_compare_K_avg_reward.png
    23_compare_K_summary_table.png
    23_compare_K_summary.txt
    23_compare_K_summary.md

Run:
    python compare_k.py
"""

import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_ROOT = os.path.join(HERE, "deep_dialog", "checkpoints", "sweep_k")
OUT = os.path.join(HERE, "slides_figures")
os.makedirs(OUT, exist_ok=True)

COLORS = ["#c0392b", "#e67e22", "#16a085", "#2c5fb0", "#5b2c6f", "#7f3b08", "#444"]


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


def plot_metric_overlay(runs_data, metric_key, title, ylabel, fname,
                        smooth_window=5, ymin=None, ymax=None, hline=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (k, rec) in enumerate(sorted(runs_data.items())):
        x = rec["x"]
        y = rec[metric_key]
        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, color=color, lw=0.9, alpha=0.35)
        ax.plot(x, smooth(y, smooth_window), color=color, lw=2.0,
                label="K = %d" % k)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if hline is not None:
        ax.axhline(hline, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def build_summary(runs_data):
    """Per-K aggregate stats."""
    rows = []
    for k, rec in sorted(runs_data.items()):
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


def write_summary_text(rows):
    txt_path = os.path.join(OUT, "23_compare_K_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DDQ planning_steps sweep — CamRest676\n")
        f.write("=" * 78 + "\n")
        f.write("%-4s %-12s %-12s %-18s %-18s %-18s\n" %
                ("K", "peak_SR", "peak_ep",
                 "final_SR (mean20)", "final_turns (m20)", "final_reward (m20)"))
        f.write("-" * 78 + "\n")
        for r in rows:
            f.write("%-4d %-12.4f %-12d %-18.4f %-18.4f %-18.4f\n" %
                    (r["K"], r["peak_sr"], r["peak_ep"],
                     r["final_sr_mean20"], r["final_turns_mean20"],
                     r["final_reward_mean20"]))
    print("saved", txt_path)

    md_path = os.path.join(OUT, "23_compare_K_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DDQ planning_steps sweep — CamRest676\n\n")
        f.write("| K | Peak SR | Peak epoch | Final SR (mean of last 20) "
                "| Final avg turns (m20) | Final avg reward (m20) |\n")
        f.write("|---|---------|------------|----------------------------"
                "|-----------------------|------------------------|\n")
        for r in rows:
            f.write("| %d | %.4f | %d | %.4f | %.2f | %.2f |\n" %
                    (r["K"], r["peak_sr"], r["peak_ep"],
                     r["final_sr_mean20"], r["final_turns_mean20"],
                     r["final_reward_mean20"]))
    print("saved", md_path)


def plot_summary_table(rows):
    fig, ax = plt.subplots(figsize=(9.5, 0.6 + 0.45 * (len(rows) + 1)))
    ax.axis("off")
    headers = ["K", "Peak SR", "Peak ep",
               "Final SR (m20)", "Final turns (m20)", "Final reward (m20)"]
    cells = []
    for r in rows:
        cells.append([
            "%d" % r["K"],
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
    ax.set_title("DDQ planning_steps sweep — summary",
                 fontsize=12, pad=8)
    p = os.path.join(OUT, "23_compare_K_summary_table.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def plot_peak_sr_bar(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ks = [r["K"] for r in rows]
    peaks = [r["peak_sr"] for r in rows]
    bars = ax.bar([str(k) for k in ks], peaks, color="#c0392b", alpha=0.85,
                  edgecolor="black")
    for b, v in zip(bars, peaks):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.2f" % v,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(0.05 + max(peaks), 1.0))
    ax.set_xlabel("planning_steps K")
    ax.set_ylabel("Peak success rate")
    ax.set_title("Peak success rate vs K — DDQ on CamRest676")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "24_compare_K_peak_sr_bar.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def main():
    runs = discover_runs()
    if not runs:
        print("No sweep results found under", SWEEP_ROOT)
        print("Run: python sweep_k.py  first.")
        return
    print("Discovered runs (K -> path):")
    for k, p in sorted(runs.items()):
        print("  K=%d  %s" % (k, p))

    runs_data = {k: load_one(p) for k, p in runs.items()}

    plot_metric_overlay(runs_data, "success_rate",
                        "Success rate by training episode (overlay) — different K",
                        "Success rate", "20_compare_K_success_rate.png",
                        ymin=-0.02, ymax=1.02)
    plot_metric_overlay(runs_data, "ave_turns",
                        "Average dialogue length by episode — different K",
                        "Average turns / dialogue",
                        "21_compare_K_avg_turns.png")
    plot_metric_overlay(runs_data, "ave_reward",
                        "Average reward by episode — different K",
                        "Average reward",
                        "22_compare_K_avg_reward.png",
                        hline=0.0)

    rows = build_summary(runs_data)
    write_summary_text(rows)
    plot_summary_table(rows)
    plot_peak_sr_bar(rows)


if __name__ == "__main__":
    main()
