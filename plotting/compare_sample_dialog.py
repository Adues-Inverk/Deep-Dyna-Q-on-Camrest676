"""
Hiển thị so sánh hội thoại mẫu trước/sau huấn luyện DQN (dạng câu đối thoại).

Đọc trường `sample_dialog_ab` trong file JSON do `run_camrest.py --agt 9` ghi.

Seed kịch bản hội thoại mẫu được chọn lúc huấn luyện (không phải trong script này):
  - Cố định:  python run_camrest.py --agt 9 --sample_dialog_seed 424242
  - Ngẫu nhiên: python run_camrest.py --agt 9 --sample_dialog_seed -1
Sau đó chạy: python compare_sample_dialog.py --result_file <path_to_json>
Tiêu đề đồ thị hiển thị rng_seed đã lưu trong JSON.
"""

import argparse
import copy
import json
import os

import matplotlib.pyplot as plt

from deep_dialog.nlg import nlg

_DEFAULT_PAIRS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'deep_dialog', 'data', 'camrest676', 'dia_act_nl_pairs_camrest.json',
)


def load_nlg_model(pairs_path=None):
    model = nlg()
    path = pairs_path or _DEFAULT_PAIRS
    if os.path.isfile(path):
        model.load_predefine_act_nl_pairs(path)
    return model


def summarize_goal(goal):
    """Mô tả ngắn mục tiêu người dùng bằng tiếng Việt."""
    if not goal:
        return '(không có goal)'
    parts = []
    for slot, val in (goal.get('inform_slots') or {}).items():
        parts.append('%s: %s' % (slot, val))
    req = list((goal.get('request_slots') or {}).keys())
    if req:
        parts.append('muốn hỏi: ' + ', '.join(req))
    return '; '.join(parts) if parts else json.dumps(goal, ensure_ascii=False)


def format_turns_as_conversation(goal, history, nlg_model):
    """
    Chuyển lịch sử dia-act (state_tracker) thành đoạn hội thoại dạng câu.
    history: list dict với speaker, diaact, inform_slots, request_slots.
    """
    lines = [
        '── Mục tiêu người dùng ──',
        summarize_goal(goal),
        '',
        '── Hội thoại ──',
    ]
    for h in history or []:
        speaker = h.get('speaker', '')
        if speaker == 'agent':
            role = 'Hệ thống'
            turn_msg = 'agt'
        else:
            role = 'Người dùng'
            turn_msg = 'usr'
        dia_act = {
            'diaact': h.get('diaact'),
            'inform_slots': copy.deepcopy(h.get('inform_slots') or {}),
            'request_slots': copy.deepcopy(h.get('request_slots') or {}),
        }
        utterance = nlg_model.convert_diaact_to_nl(dia_act, turn_msg)
        lines.append('%s: %s' % (role, utterance))
    return '\n'.join(lines)


def format_outcome_footer(reward, turn_count, success):
    return (
        '── Kết quả ──\n'
        'Thành công: %s | Phần thưởng: %s | Số lượt: %s'
        % ('có' if success else 'không', reward, turn_count)
    )


def resolve_conversation_text(block, side, nlg_model):
    """
    side: 'before' | 'after'
    Ưu tiên bản conversation đã lưu; nếu không có thì dựng từ goal + turns.
    """
    conv_key = 'before_rl_conversation' if side == 'before' else 'after_best_conversation'
    if block.get(conv_key):
        return block[conv_key]

    turns_key = 'before_turns' if side == 'before' else 'after_turns'
    turns = block.get(turns_key)
    goal = block.get('goal')
    if turns is not None and goal is not None:
        body = format_turns_as_conversation(goal, turns, nlg_model)
        outcome = block.get('%s_outcome' % side)
        if outcome:
            body += '\n\n' + format_outcome_footer(
                outcome.get('reward', '?'),
                outcome.get('turn_count', '?'),
                outcome.get('success', False),
            )
        return body

    legacy_key = 'before_rl_episodes' if side == 'before' else 'after_best_policy'
    legacy = block.get(legacy_key) or ''
    if legacy and ' | inform=' in legacy:
        return (
            '(Dữ liệu cũ dạng dia-act. Chạy lại huấn luyện:\n'
            '  python run_camrest.py --agt 9 ...\n'
            'để ghi transcript dạng hội thoại vào JSON.)\n\n' + legacy
        )
    return legacy


def _wrap_text_block(text, width=88):
    lines_out = []
    for line in text.split('\n'):
        while len(line) > width:
            lines_out.append(line[:width])
            line = line[width:]
        lines_out.append(line)
    return '\n'.join(lines_out)


def draw_sample_dialog_compare(path, pairs_path=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    block = data.get('sample_dialog_ab')
    if not block:
        print(
            'File JSON không có trường sample_dialog_ab. '
            'Chạy huấn luyện DQN: python run_camrest.py --agt 9 ...'
        )
        return

    nlg_model = load_nlg_model(pairs_path)
    before = _wrap_text_block(resolve_conversation_text(block, 'before', nlg_model))
    after = _wrap_text_block(resolve_conversation_text(block, 'after', nlg_model))
    seed = block.get('rng_seed', '?')
    best_ep = block.get('best_rl_epoch', '?')

    nlines = max(before.count('\n'), after.count('\n')) + 1
    fig_h = min(26, max(8, 0.16 * nlines))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, fig_h))
    fig.suptitle(
        'So sánh hội thoại mẫu (seed RNG = %s) | epoch RL tốt nhất: %s'
        % (seed, best_ep),
        fontsize=12,
    )
    for ax, title, txt in (
        (ax1, 'Trước các episode RL', before),
        (ax2, 'Sau huấn luyện (best policy)', after),
    ):
        ax.axis('off')
        ax.set_title(title, fontsize=11, pad=10)
        ax.text(
            0.02, 0.98, txt or '(trống)',
            transform=ax.transAxes,
            fontsize=7.5,
            family='sans-serif',
            va='top', ha='left',
        )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            'So sánh hội thoại mẫu trước/sau huấn luyện (đọc từ JSON). '
            'Để dùng seed ngẫu nhiên cho kịch bản mẫu, huấn luyện với '
            'run_camrest.py --agt 9 --sample_dialog_seed -1 trước.'
        ),
    )
    parser.add_argument(
        '--result_file',
        type=str,
        default='./deep_dialog/checkpoints/agt_9_performance_records.json',
    )
    parser.add_argument(
        '--diaact_nl_pairs',
        type=str,
        default='',
        help='Đường dẫn dia_act_nl_pairs_camrest.json (mặc định: trong deep_dialog/data/...)',
    )
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    pairs = args.diaact_nl_pairs.strip() or None
    draw_sample_dialog_compare(args.result_file, pairs_path=pairs)


if __name__ == '__main__':
    main()
