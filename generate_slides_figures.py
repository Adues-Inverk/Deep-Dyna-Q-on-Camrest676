"""
Sinh cac hinh minh hoa cho bai thuyet trinh Deep Dyna-Q.

Luu tat ca file PNG vao ./slides_figures/.

Chay:
    python generate_slides_figures.py
"""

import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
# 1) Learning curves tu ket qua huan luyen thuc te
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
    ax.set_xlabel("Episode huan luyen")
    ax.set_ylabel("Success Rate")
    ax.set_title("Duong cong hoc tap DDQ - Success Rate tren CamRest676")
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
    ax.set_xlabel("Episode huan luyen")
    ax.set_ylabel("So luot trung binh moi hoi thoai")
    ax.set_title("Average dialogue length giam dan - bang chung step-penalty hoat dong")
    ax.grid(True, alpha=0.4)
    ax.axhline(y=np.mean(at[-20:]), color="green", linestyle="--", lw=1,
               label="Trung binh 20 epoch cuoi = %.1f" % np.mean(at[-20:]))
    ax.legend()
    fig.tight_layout()
    savefig(fig, "02_avg_turns_curve.png")


def plot_avg_reward(x, ar):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, ar, color="#16a085", lw=1.6)
    ax.axhline(y=0, color="black", linestyle=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Episode huan luyen")
    ax.set_ylabel("Phan thuong trung binh")
    ax.set_title("Avg Reward - DDQ vuot nguong 0 sau warm-start")
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
    fig.suptitle("DDQ tren CamRest676 - tong quan learning curves", y=1.03)
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
    ax.set_xlabel("So luot trong mot hoi thoai")
    ax.set_ylabel("So hoi thoai")
    ax.set_title("Phan phoi do dai hoi thoai - epoch %d (n=%d)" % (best_key, len(turns)))
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
    bars1 = ax.bar(x - w/2, before, w, label="Truoc RL (random policy)", color="#7f8c8d")
    bars2 = ax.bar(x + w/2, after, w, label="Sau RL (best @ ep %s)" % a.get("best_rl_epoch","?"),
                   color="#e67e22")
    for bars in (bars1, bars2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + (0.5 if h >= 0 else -1.5),
                    "%.2f" % h, ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("So sanh A/B - n=%s hoi thoai danh gia" % block.get("eval_episodes","?"))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "07_before_after_metrics.png")


# -------------------------------------------------------------------
# 4) Conceptual figures
# -------------------------------------------------------------------
def draw_arrow(ax, p1, p2, color="black", lw=1.3, style="->", curve=0.0):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                          lw=lw, mutation_scale=12,
                          connectionstyle="arc3,rad=%f" % curve)
    ax.add_patch(arr)


def box(ax, xy, w, h, text, color="#dfe6f5", edge="#1f4068", fontsize=10):
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                          fc=color, ec=edge, lw=1.6)
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text,
            ha="center", va="center", fontsize=fontsize)


def plot_ddq_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("Kien truc Deep Dyna-Q: Direct RL + World Model + Planning",
                 fontsize=13, pad=10)

    box(ax, (0.3, 4.5), 2.4, 1.0, "User\nSimulator", color="#fdebd0", edge="#b9770e")
    box(ax, (4.3, 4.5), 2.4, 1.0, "Q-Network\n(Dialogue Policy)", color="#d1f2eb", edge="#117864")
    box(ax, (8.3, 4.5), 2.4, 1.0, "Real Experience\nBuffer D_real",
        color="#dfe6f5", edge="#1f4068")
    box(ax, (0.3, 1.5), 2.4, 1.0, "World Model\nM(s, a; theta_M)", color="#fadbd8", edge="#922b21")
    box(ax, (4.3, 1.5), 2.4, 1.0, "Planning\n(K-step rollouts)", color="#e8daef", edge="#5b2c6f")
    box(ax, (8.3, 1.5), 2.4, 1.0, "Simulated\nBuffer D_sim",
        color="#dfe6f5", edge="#1f4068")

    draw_arrow(ax, (2.7, 5.0), (4.3, 5.0))
    ax.text(3.5, 5.15, "state s", fontsize=9, ha="center")
    draw_arrow(ax, (4.3, 4.8), (2.7, 4.8))
    ax.text(3.5, 4.6, "action a", fontsize=9, ha="center")
    draw_arrow(ax, (6.7, 5.0), (8.3, 5.0))
    ax.text(7.5, 5.15, "(s,a,r,s')", fontsize=9, ha="center")
    draw_arrow(ax, (9.5, 4.5), (9.5, 2.5), curve=-0.2)
    ax.text(10.2, 3.5, "train\nM, Q", fontsize=9, ha="center")
    draw_arrow(ax, (8.3, 2.0), (6.7, 2.0))
    ax.text(7.5, 2.2, "sample", fontsize=9, ha="center")
    draw_arrow(ax, (4.3, 2.0), (2.7, 2.0))
    ax.text(3.5, 2.2, "query M(s,a)", fontsize=9, ha="center")
    draw_arrow(ax, (2.7, 1.8), (4.3, 1.8))
    ax.text(3.5, 1.6, "(s',r,t)", fontsize=9, ha="center")
    draw_arrow(ax, (5.5, 2.5), (5.5, 4.5))
    ax.text(5.9, 3.5, "gradient\nupdate", fontsize=9, ha="left")

    ax.text(5.5, 0.4,
            "Direct RL (D_real) + Model-based Planning (D_sim) => tang sample efficiency",
            ha="center", fontsize=10, style="italic", color="#444")
    fig.tight_layout()
    savefig(fig, "08_ddq_architecture.png")


def plot_three_pillars():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Ba tru cot cua Deep Dyna-Q", fontsize=13, pad=10)

    pillars = [
        ("Direct RL",
         "Tuong tac THAT voi\nuser simulator.\nCap nhat Q-Network\ntruc tiep tu D_real.",
         "#d1f2eb", "#117864"),
        ("World Model\nLearning",
         "Hoc p(s', r, t | s, a)\nbang supervised\nlearning tren D_real.",
         "#fadbd8", "#922b21"),
        ("Planning",
         "Sinh hoi thoai MO PHONG\ntu M(s,a). K buoc\ntren moi buoc Direct RL.",
         "#e8daef", "#5b2c6f"),
    ]
    for i, (title, body, fc, ec) in enumerate(pillars):
        x0 = 0.4 + i * 3.2
        box(ax, (x0, 1.0), 2.8, 2.6, "", color=fc, edge=ec)
        ax.text(x0 + 1.4, 3.15, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x0 + 1.4, 1.95, body, ha="center", va="center", fontsize=10)
        ax.text(x0 + 1.4, 0.55, "Pillar %d" % (i+1), ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    savefig(fig, "09_three_pillars.png")


def plot_kstep_planning_schematic():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Vong lap Planning K buoc (1 buoc Direct RL => K buoc mo phong)",
                 fontsize=12, pad=8)

    box(ax, (0.3, 2.6), 2.0, 1.0, "Real step\n(s,a,r,s')", color="#d1f2eb", edge="#117864")
    ax.text(1.3, 2.4, "Direct RL", ha="center", fontsize=9, color="#117864")
    draw_arrow(ax, (2.3, 3.1), (3.4, 3.1))

    for k, x0 in enumerate([3.4, 5.4, 7.4]):
        box(ax, (x0, 2.6), 1.8, 1.0,
            "Sim step %d\n(M generates\ns',r,t)" % (k+1),
            color="#e8daef", edge="#5b2c6f", fontsize=9)
        if k < 2:
            draw_arrow(ax, (x0 + 1.8, 3.1), (x0 + 2.0, 3.1))
    ax.text(9.3, 3.1, "...", fontsize=14, va="center")

    box(ax, (3.4, 0.6), 5.8, 1.0,
        "Moi (s,a,r,s') noi vao D_sim => cap nhat Q-Network",
        color="#dfe6f5", edge="#1f4068", fontsize=10)
    draw_arrow(ax, (6.3, 2.55), (6.3, 1.6))
    fig.tight_layout()
    savefig(fig, "10_kstep_planning.png")


def plot_sample_efficiency():
    rng = np.random.default_rng(42)
    eps = np.arange(0, 600)
    def sigmoid_curve(t, mid, scale, top=0.9, noise=0.04):
        s = top / (1 + np.exp(-(t - mid) / scale))
        s = s + rng.normal(0, noise, size=t.shape)
        return np.clip(s, 0, 1)
    ddq = sigmoid_curve(eps, mid=130, scale=25, top=0.92, noise=0.025)
    dqn = sigmoid_curve(eps, mid=420, scale=70, top=0.90, noise=0.025)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eps, ddq, color="#c0392b", lw=1.8, label="DDQ (K=5)")
    ax.plot(eps, dqn, color="#2c5fb0", lw=1.8, label="DQN (model-free)")
    ax.axhline(0.9, color="green", ls="--", lw=1, label="Nguong 90% success")
    ax.axvline(150, color="grey", ls=":", lw=1)
    ax.text(155, 0.2, "DDQ cham 90%\n@ ~150 eps", fontsize=9, color="#c0392b")
    ax.axvline(500, color="grey", ls=":", lw=1)
    ax.text(505, 0.2, "DQN cham 90%\n@ ~500 eps", fontsize=9, color="#2c5fb0")
    ax.set_xlabel("Episode huan luyen")
    ax.set_ylabel("Success Rate")
    ax.set_title("Sample Efficiency - DDQ vs DQN (minh hoa dinh tinh)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    savefig(fig, "11_sample_efficiency_illustration.png")


def plot_compounding_error():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    k = 1.7
    steps = np.arange(0, 8)
    eps0 = 0.05
    err = eps0 * (k ** steps)
    ax.bar(steps, err, color="#c0392b", alpha=0.85, edgecolor="black")
    for s, e in zip(steps, err):
        ax.text(s, e, "%.2f" % e, ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("So buoc planning (K)")
    ax.set_ylabel("Sai so tich luy")
    ax.set_title("Compounding Error: eps -> k*eps -> k^2*eps -> ... (k=1.7, eps0=0.05)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "12_compounding_error.png")


def plot_reward_design():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    turns = np.arange(1, 31)
    L = 30
    succ = 2 * L - turns
    fail = -L - turns
    ax.plot(turns, succ, marker="o", color="#16a085", lw=1.6, label="Hoan thanh: +2L - turns")
    ax.plot(turns, fail, marker="o", color="#c0392b", lw=1.6, label="That bai: -L - turns")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Do dai hoi thoai (turns)")
    ax.set_ylabel("Phan thuong ky vong cuoi")
    ax.set_title("Reward design: step-penalty -1 => uu tien hoi thoai ngan")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "13_reward_design.png")


def plot_pipeline_tods():
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("Pipeline truyen thong cua he thong hoi thoai huong tac vu",
                 fontsize=12, pad=8)
    modules = [
        ("NLU", "Intent + Slots", "#fdebd0", "#b9770e"),
        ("DST", "Belief state\ntracker", "#fadbd8", "#922b21"),
        ("Dialogue\nPolicy", "Chon action\n(DDQ o day!)", "#d1f2eb", "#117864"),
        ("NLG", "Sinh van ban\ntra loi", "#dfe6f5", "#1f4068"),
    ]
    for i, (t, b, fc, ec) in enumerate(modules):
        x0 = 0.5 + i * 2.6
        box(ax, (x0, 0.9), 2.0, 1.6, "", color=fc, edge=ec)
        ax.text(x0 + 1.0, 1.95, t, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(x0 + 1.0, 1.25, b, ha="center", va="center", fontsize=9)
        if i < 3:
            draw_arrow(ax, (x0 + 2.0, 1.7), (x0 + 2.6, 1.7))
    fig.tight_layout()
    savefig(fig, "14_tod_pipeline.png")


def plot_d3q_discriminator():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("D3Q - bo sung mang Discriminator loc du lieu mo phong",
                 fontsize=12, pad=10)
    box(ax, (0.3, 2.7), 2.0, 1.0, "World Model\nM(s,a)", color="#fadbd8", edge="#922b21")
    box(ax, (3.0, 2.7), 2.0, 1.0, "Sim. sample\n(s,a,r,s')", color="#e8daef", edge="#5b2c6f")
    box(ax, (5.7, 2.7), 2.0, 1.0, "Discriminator\nD_phi(x)", color="#fdebd0", edge="#b9770e")
    box(ax, (8.4, 2.7), 1.5, 1.0, "D_sim\n(filtered)", color="#dfe6f5", edge="#1f4068")
    draw_arrow(ax, (2.3, 3.2), (3.0, 3.2))
    draw_arrow(ax, (5.0, 3.2), (5.7, 3.2))
    draw_arrow(ax, (7.7, 3.2), (8.4, 3.2))
    ax.text(7.05, 3.55, "D_phi(x) > tau ?", fontsize=9, color="#b9770e", ha="center")
    box(ax, (5.7, 0.8), 2.0, 1.0, "D_real\n(label 1)", color="#d1f2eb", edge="#117864")
    draw_arrow(ax, (6.7, 1.8), (6.7, 2.7))
    ax.text(7.6, 2.3, "label 0 vs 1\nbinary CE", fontsize=9, color="#444")
    fig.tight_layout()
    savefig(fig, "15_d3q_discriminator.png")


def plot_active_ddq_switch():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Switch-based Active DDQ - Q-Ensemble do do bat dinh",
                 fontsize=12, pad=10)
    box(ax, (0.3, 2.7), 2.2, 1.2, "Q-Ensemble\n{Q1, Q2, ..., QN}",
        color="#d1f2eb", edge="#117864")
    box(ax, (3.4, 2.7), 2.2, 1.2, "Var(Q) - Do\nbat dinh", color="#fadbd8", edge="#922b21")
    box(ax, (6.6, 4.0), 2.6, 0.7, "Var(Q) > eps => Hoi Human",
        color="#e8daef", edge="#5b2c6f", fontsize=9)
    box(ax, (6.6, 2.7), 2.6, 0.7, "Var(Q) <= eps => Plan voi M",
        color="#e8daef", edge="#5b2c6f", fontsize=9)
    draw_arrow(ax, (2.5, 3.3), (3.4, 3.3))
    draw_arrow(ax, (5.6, 3.5), (6.6, 4.3), curve=0.2)
    draw_arrow(ax, (5.6, 3.0), (6.6, 3.0), curve=-0.2)
    ax.text(0.3, 1.7, "Y tuong: chi truy van nguon du lieu dat khi mo hinh thieu chac chan.",
            fontsize=10, style="italic", color="#444")
    fig.tight_layout()
    savefig(fig, "16_active_ddq_switch.png")


def plot_budgeted_ddq():
    rng = np.random.default_rng(7)
    t = np.arange(0, 200)
    n_r = np.clip(20 - 0.08 * t + rng.normal(0, 0.4, t.shape), 1, 20)
    K = np.clip(2 + 0.10 * t + rng.normal(0, 0.6, t.shape), 1, 30)

    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(t, n_r, color="#c0392b", lw=1.8, label="Tuong tac that (n_r)")
    l2 = ax2.plot(t, K, color="#2c5fb0", lw=1.8, label="So buoc planning K")
    ax1.set_xlabel("Vong huan luyen t")
    ax1.set_ylabel("Tuong tac that (n_r)", color="#c0392b")
    ax2.set_ylabel("So buoc planning K", color="#2c5fb0")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#2c5fb0")
    ax1.set_title("Budgeted DDQ - giam tuong tac that, tang planning theo ngan sach B")
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center")
    fig.tight_layout()
    savefig(fig, "17_budgeted_ddq.png")


def plot_mdp_loop():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Hoi thoai nhu MDP <S, A, R, gamma>", fontsize=12, pad=10)
    box(ax, (0.5, 2.3), 2.5, 1.3, "Agent\n(Dialogue Policy)",
        color="#d1f2eb", edge="#117864")
    box(ax, (5.5, 2.3), 2.5, 1.3, "Environment\n(User + KB)",
        color="#fadbd8", edge="#922b21")
    draw_arrow(ax, (3.0, 3.1), (5.5, 3.1))
    ax.text(4.25, 3.3, "action a_t", fontsize=10, ha="center")
    draw_arrow(ax, (5.5, 2.6), (3.0, 2.6), curve=-0.3)
    ax.text(4.25, 1.85, "state s_{t+1}, reward r_t", fontsize=10, ha="center")
    ax.text(4.25, 0.8,
            "r_t = +L (success) | -L (fail) | -1 (moi turn)",
            ha="center", fontsize=10, style="italic", color="#444")
    fig.tight_layout()
    savefig(fig, "18_mdp_loop.png")


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

    print("=> tat ca anh da luu vao", OUT)


if __name__ == "__main__":
    main()
