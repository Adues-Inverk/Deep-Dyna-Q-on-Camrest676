"""
Generate illustration figures for the Deep Dyna-Q presentation.

Saves all PNGs to ./slides_figures/.

Run:
    python generate_slides_figures.py
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "slides_figures")
PERF = os.path.join(HERE, "deep_dialog", "checkpoints", "agt_9_performance_records.json")

os.makedirs(OUT, exist_ok=True)


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# -------------------------------------------------------------------
# 1) Learning curves from actual training results
# -------------------------------------------------------------------
def load_perf():
    with open(PERF, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = sorted(int(k) for k in data["success_rate"].keys() if int(k) >= 0)
    x = keys
    sr = [data["success_rate"][str(k)] for k in keys]
    at = [data["ave_turns"][str(k)] for k in keys]
    ar = [data["ave_reward"][str(k)] for k in keys]
    bl = data.get("bellman_loss", {})
    loss = [float(bl[str(k)]) if str(k) in bl else float("nan") for k in keys]
    return x, sr, at, ar, loss, data


def plot_success_rate(x, sr):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, sr, color="#c0392b", lw=1.8)
    ax.fill_between(x, sr, alpha=0.15, color="#c0392b")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("DDQ learning curve - Success Rate on CamRest676")
    ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.02, 1.02)
    best_ep = int(np.argmax(sr))
    ax.scatter([x[best_ep]], [sr[best_ep]], color="black", zorder=5, s=40)
    ax.annotate("Peak %.0f%% @ ep %d" % (sr[best_ep]*100, x[best_ep]),
                xy=(x[best_ep], sr[best_ep]),
                xytext=(x[best_ep]+5, sr[best_ep]-0.12),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.tight_layout()
    savefig(fig, "01_success_rate_curve.png")


def plot_avg_turns(x, at):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, at, color="#2c5fb0", lw=1.6)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average turns per dialogue")
    ax.set_title("Average dialogue length decreases - evidence that step-penalty works")
    ax.grid(True, alpha=0.4)
    ax.axhline(y=np.mean(at[-20:]), color="green", linestyle="--", lw=1,
               label="Mean over last 20 epochs = %.1f" % np.mean(at[-20:]))
    ax.legend()
    fig.tight_layout()
    savefig(fig, "02_avg_turns_curve.png")


def plot_avg_reward(x, ar):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, ar, color="#16a085", lw=1.6)
    ax.axhline(y=0, color="black", linestyle=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average reward")
    ax.set_title("Average reward - DDQ crosses zero after warm-start")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    savefig(fig, "03_avg_reward_curve.png")


def plot_bellman_loss(x, loss):
    finite_mask = [math.isfinite(v) for v in loss]
    if not any(finite_mask):
        return
    xs = [xi for xi, m in zip(x, finite_mask) if m]
    ys = [v for v, m in zip(loss, finite_mask) if m]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, ys, color="#7f3b08", lw=1.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("TD MSE (Bellman error)")
    ax.set_title("DQN Bellman / TD-target Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    savefig(fig, "04_bellman_loss_curve.png")


def plot_combined_curves(x, sr, at, ar, loss):
    has_loss = any(math.isfinite(v) for v in loss)
    n = 4 if has_loss else 3
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), sharex=True)
    axes[0].plot(x, sr, color="#c0392b", lw=1.6)
    axes[0].set_title("Success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(x, at, color="#2c5fb0", lw=1.6)
    axes[1].set_title("Avg turns")
    axes[2].plot(x, ar, color="#16a085", lw=1.6)
    axes[2].set_title("Avg reward")
    axes[2].axhline(0, color="black", ls=":", lw=0.7)
    for a in axes[:3]:
        a.grid(True, alpha=0.4)
        a.set_xlabel("Episode")
    if has_loss:
        ys = [v if math.isfinite(v) else float("nan") for v in loss]
        axes[3].plot(x, ys, color="#7f3b08", lw=1.6)
        axes[3].set_title("Bellman loss")
        axes[3].grid(True, alpha=0.4)
        axes[3].set_xlabel("Episode")
    fig.suptitle("DDQ on CamRest676 - learning curves overview", y=1.03)
    fig.tight_layout()
    savefig(fig, "05_combined_learning_curves.png")


# -------------------------------------------------------------------
# 2) Turn distribution histogram
# -------------------------------------------------------------------

def plot_turn_histogram(data):
    et = data.get("eval_dialog_turns")
    if not et:
        return
    keys = sorted(int(k) for k in et.keys())
    sr = data["success_rate"]
    best_key = max(keys, key=lambda k: sr.get(str(k), 0))
    turns = [int(t) for t in et[str(best_key)]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    counts, bins, patches_ = ax.hist(turns, bins="auto", color="#3a78b8",
                                     edgecolor="black", alpha=0.85)
    for c, p in zip(counts, patches_):
        if c > 0:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height(),
                    "%d" % int(c), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Turns per dialogue")
    ax.set_ylabel("Number of dialogues")
    ax.set_title("Dialogue length distribution - epoch %d (n=%d)" % (best_key, len(turns)))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "06_turn_distribution.png")


# -------------------------------------------------------------------
# 3) Before vs After metrics
# -------------------------------------------------------------------
def plot_metrics_before_after(data):
    block = data.get("metrics_ab")
    if not block:
        return
    b, a = block["before_rl"], block["after_best_policy"]
    labels = ["Success rate", "Avg turns", "Avg reward"]
    before = [b["success_rate"], b["ave_turns"], b["ave_reward"]]
    after = [a["success_rate"], a["ave_turns"], a["ave_reward"]]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars1 = ax.bar(x - w/2, before, w, label="Before RL (random policy)", color="#7f8c8d")
    bars2 = ax.bar(x + w/2, after, w, label="After RL (best @ ep %s)" % a.get("best_rl_epoch","?"),
                   color="#e67e22")
    for bars in (bars1, bars2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + (0.5 if h >= 0 else -1.5),
                    "%.2f" % h, ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("A/B comparison - n=%s evaluation dialogues" % block.get("eval_episodes","?"))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "07_before_after_metrics.png")


def main():
    x, sr, at, ar, loss, data = load_perf()
    print("loaded %d episodes; success_rate in [%.2f, %.2f]; avg_turns in [%.1f, %.1f]"
          % (len(x), min(sr), max(sr), min(at), max(at)))

    plot_success_rate(x, sr)
    plot_avg_turns(x, at)
    plot_avg_reward(x, ar)
    plot_bellman_loss(x, loss)
    plot_combined_curves(x, sr, at, ar, loss)
    plot_turn_histogram(data)
    plot_metrics_before_after(data)

    print("=> all figures saved to", OUT)


if __name__ == "__main__":
    main()
