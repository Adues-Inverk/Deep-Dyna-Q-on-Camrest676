"""
Compare k=0 (no planning) vs k=7 (best k) from
deep_dialog/checkpoints/compare/k_{0,7}/agt_9_performance_records.json

Outputs to slides_figures/:
  compare_k0_k7_success_rate.png
  compare_k0_k7_avg_turns.png
  compare_k0_k7_avg_reward.png
  compare_k0_k7_summary_table.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
CMP    = os.path.join(ROOT, "deep_dialog", "checkpoints", "compare")
OUT    = os.path.join(HERE, "slides_figures")
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "k=0\n(no planning)": os.path.join(CMP, "k_0", "agt_9_performance_records.json"),
    "k=7\n(best k)":       os.path.join(CMP, "k_7", "agt_9_performance_records.json"),
}
COLORS = {"k=0\n(no planning)": "#e74c3c", "k=7\n(best k)": "#2980b9"}


def load(path):
    with open(path) as f:
        d = json.load(f)
    eps = sorted(d["success_rate"].keys(), key=int)
    return {
        "x":  [int(e) for e in eps],
        "sr": [d["success_rate"][e] for e in eps],
        "at": [d["ave_turns"][e]    for e in eps],
        "ar": [d["ave_reward"][e]   for e in eps],
        "raw": d,
    }


def smooth(y, w=15):
    y = np.asarray(y, dtype=float)
    w = max(1, min(w, len(y)))
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot_metric(data, key, ylabel, title, fname, ymin=None, ymax=None, hline=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, rec in data.items():
        c = COLORS[label]
        y = rec[key]
        ax.plot(rec["x"], y, color=c, lw=0.8, alpha=0.25)
        ax.plot(rec["x"], smooth(y), color=c, lw=2.2, label=label.replace("\n", " "))
    ax.set_xlabel("Training episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.35)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if hline is not None:
        ax.axhline(hline, color="black", ls=":", lw=0.9, alpha=0.5)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def plot_summary(rows):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    headers = ["Model", "Peak SR", "Peak epoch",
               "Final SR (mean last 20)", "Final avg turns", "Final avg reward"]
    cells = [[r["label"], "%.3f" % r["peak_sr"], str(r["peak_ep"]),
              "%.3f ± %.3f" % (r["mean20_sr"], r["std20_sr"]),
              "%.2f" % r["mean20_at"], "%.2f" % r["mean20_ar"]] for r in rows]
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#cce0f5")
        tbl[(0, j)].set_text_props(weight="bold")
    # highlight better value per numeric column (columns 1,3,4,5)
    for col_idx, better in [(1, max), (3, max), (4, min), (5, max)]:
        vals = [float(cells[r][col_idx].split()[0]) for r in range(len(rows))]
        best = better(vals)
        for r, v in enumerate(vals):
            if v == best:
                tbl[(r + 1, col_idx)].set_facecolor("#d5f5e3")
    ax.set_title("k=0 vs k=7 — CamRest676 DQN (300 episodes)", fontsize=12, pad=8)
    p = os.path.join(OUT, "compare_k0_k7_summary_table.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def main():
    data = {}
    for label, path in RUNS.items():
        if not os.path.isfile(path):
            print("Missing:", path)
            sys.exit(1)
        data[label] = load(path)

    plot_metric(data, "sr", "Success rate",
                "Success rate — k=0 vs k=7 (300 episodes, CamRest676)",
                "compare_k0_k7_success_rate.png", ymin=-0.02, ymax=1.02)
    plot_metric(data, "at", "Average turns / dialogue",
                "Average dialogue length — k=0 vs k=7",
                "compare_k0_k7_avg_turns.png")
    plot_metric(data, "ar", "Average reward",
                "Average reward — k=0 vs k=7",
                "compare_k0_k7_avg_reward.png", hline=0)

    rows = []
    for label, rec in data.items():
        sr = rec["sr"]
        at = rec["at"]
        ar = rec["ar"]
        best_ep_idx = int(np.argmax(sr))
        last20_sr = sr[-20:]
        last20_at = at[-20:]
        last20_ar = ar[-20:]
        rows.append({
            "label":    label.replace("\n", " "),
            "peak_sr":  max(sr),
            "peak_ep":  rec["x"][best_ep_idx],
            "mean20_sr": float(np.mean(last20_sr)),
            "std20_sr":  float(np.std(last20_sr)),
            "mean20_at": float(np.mean(last20_at)),
            "mean20_ar": float(np.mean(last20_ar)),
        })
    plot_summary(rows)

    print("\n=== Summary ===")
    for r in rows:
        print(f"  {r['label']}")
        print(f"    Peak SR:          {r['peak_sr']:.3f}  (ep {r['peak_ep']})")
        print(f"    Final SR mean20:  {r['mean20_sr']:.3f} ± {r['std20_sr']:.3f}")
        print(f"    Final avg turns:  {r['mean20_at']:.2f}")
        print(f"    Final avg reward: {r['mean20_ar']:.2f}")


if __name__ == "__main__":
    main()
