"""
In bảng / biểu đồ so sánh metrics trước và sau huấn luyện DQN.

Đọc trường `metrics_ab` trong JSON do `run_camrest.py --agt 9` ghi.

  python compare_metrics.py --result_file deep_dialog/checkpoints/agt_9_performance_records.json
  python compare_metrics.py --plot
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

_METRIC_LABELS = (
    ('success_rate', 'Success rate'),
    ('ave_turns', 'Avg turns'),
    ('ave_reward', 'Avg reward'),
)


def load_metrics_ab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    block = data.get('metrics_ab')
    if not block:
        return None
    return block


def print_metrics_ab(path):
    block = load_metrics_ab(path)
    if block is None:
        print(
            'File JSON không có trường metrics_ab.\n'
            'Chạy huấn luyện DQN: python run_camrest.py --agt 9 ...'
        )
        return False

    b = block['before_rl']
    a = block['after_best_policy']
    d = block['delta']
    n = block.get('eval_episodes', '?')
    best_ep = a.get('best_rl_epoch', '?')

    print('Eval episodes: %s | Best RL epoch: %s' % (n, best_ep))
    print('%-16s %12s %12s %12s' % ('', 'Before RL', 'After (best)', 'Delta'))
    print('-' * 56)
    for key, label in _METRIC_LABELS:
        print('%-16s %12.4f %12.4f %12.4f' % (label, b[key], a[key], d[key]))
    return True


def plot_metrics_ab(path):
    block = load_metrics_ab(path)
    if block is None:
        print(
            'File JSON không có trường metrics_ab.\n'
            'Chạy huấn luyện DQN: python run_camrest.py --agt 9 ...'
        )
        return

    b = block['before_rl']
    a = block['after_best_policy']
    labels = [label for _, label in _METRIC_LABELS]
    before_vals = [b[k] for k, _ in _METRIC_LABELS]
    after_vals = [a[k] for k, _ in _METRIC_LABELS]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], before_vals, width, label='Before RL', color='steelblue')
    ax.bar([i + width / 2 for i in x], after_vals, width, label='After (best policy)', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(
        'Metrics comparison (n=%s eval, best epoch %s)'
        % (block.get('eval_episodes', '?'), a.get('best_rl_epoch', '?'))
    )
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='So sánh success_rate / ave_turns / ave_reward trước và sau RL.',
    )
    parser.add_argument(
        '--result_file',
        type=str,
        default='./deep_dialog/checkpoints/agt_9_performance_records.json',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Vẽ biểu đồ cột trước/sau (cần matplotlib).',
    )
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    if not os.path.isfile(args.result_file):
        print('Không tìm thấy file: %s' % args.result_file)
        return
    if args.plot:
        plot_metrics_ab(args.result_file)
    else:
        print_metrics_ab(args.result_file)


if __name__ == '__main__':
    main()
