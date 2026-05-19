"""
Luu hinh dau ra cua compare_metrics.py va compare_sample_dialog.py
vao thu muc slides_figures/.

Khong sua hai script goc — chi monkey-patch plt.show() de savefig roi close.

Chay:
    python save_compare_outputs.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_metrics
import compare_sample_dialog


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "slides_figures")
PERF = os.path.join(HERE, "deep_dialog", "checkpoints", "agt_9_performance_records.json")

os.makedirs(OUT, exist_ok=True)


def _make_show(out_path):
    def _show(*_a, **_kw):
        fig = plt.gcf()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("saved", out_path)
    return _show


def save_metrics_bar():
    out = os.path.join(OUT, "08_compare_metrics_bar.png")
    plt.show = _make_show(out)
    compare_sample_dialog.plt.show = plt.show  # cung 1 module matplotlib
    compare_metrics.plt.show = plt.show
    compare_metrics.plot_metrics_ab(PERF)


def save_sample_dialog_compare():
    out = os.path.join(OUT, "09_sample_dialog_compare.png")
    plt.show = _make_show(out)
    compare_sample_dialog.plt.show = plt.show
    compare_metrics.plt.show = plt.show
    compare_sample_dialog.draw_sample_dialog_compare(PERF)


def save_sample_dialog_text():
    """Dump hoi thoai mau (truoc/sau RL) vao file text doc duoc."""
    with open(PERF, "r", encoding="utf-8") as f:
        data = json.load(f)
    block = data.get("sample_dialog_ab")
    if not block:
        print("(khong co sample_dialog_ab trong JSON, bo qua dump text)")
        return
    seed = block.get("rng_seed", "?")
    best_ep = block.get("best_rl_epoch", "?")
    before = block.get("before_rl_conversation") or ""
    after = block.get("after_best_conversation") or ""

    out_txt = os.path.join(OUT, "10_sample_dialog_compare.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("So sanh hoi thoai mau truoc/sau huan luyen DDQ\n")
        f.write("RNG seed: %s | Epoch RL tot nhat: %s\n" % (seed, best_ep))
        f.write("=" * 80 + "\n\n")
        f.write("###### TRUOC RL (random policy) ######\n\n")
        f.write(before + "\n\n")
        f.write("###### SAU RL (best policy) ######\n\n")
        f.write(after + "\n")
    print("saved", out_txt)

    # them ban markdown de tien chen vao slide
    out_md = os.path.join(OUT, "10_sample_dialog_compare.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# So sanh hoi thoai mau truoc/sau RL\n\n")
        f.write("- RNG seed: `%s`\n" % seed)
        f.write("- Epoch RL tot nhat: `%s`\n\n" % best_ep)
        f.write("## Truoc RL\n\n```\n%s\n```\n\n" % before)
        f.write("## Sau RL (best policy)\n\n```\n%s\n```\n" % after)
    print("saved", out_md)


def main():
    save_metrics_bar()
    save_sample_dialog_compare()
    save_sample_dialog_text()
    print("=> da luu output cua compare_metrics & compare_sample_dialog vao", OUT)


if __name__ == "__main__":
    main()
