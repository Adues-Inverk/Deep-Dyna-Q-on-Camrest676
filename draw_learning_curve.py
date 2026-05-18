'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse, json
import math

import matplotlib.pyplot as plt


def read_performance_records(path):
    """ load the performance score (.json) file """
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    has_bl = 'bellman_loss' in data
    bl = data['bellman_loss'] if has_bl else {}
    for key in data['success_rate'].keys():
        if int(key) > -1:
            tail = ("\t%s" % bl.get(str(key), '')) if has_bl else ''
            print("%s\t%s\t%s\t%s%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key], tail))
            

def load_performance_file(path):
    """ load the performance score (.json) file """
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    numbers = {'x': [], 'success_rate':[], 'ave_turns':[], 'ave_rewards':[], 'bellman_loss':[]}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()
    bl = data.get('bellman_loss', {})

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])
            if str(key) in bl:
                numbers['bellman_loss'].append(float(bl[str(key)]))
            else:
                numbers['bellman_loss'].append(float('nan'))
    return numbers

def draw_learning_curve(numbers):
    """ draw success rate, average turns, and DQN Bellman / TD MSE loss """
    has_loss = any(math.isfinite(v) for v in numbers['bellman_loss'])
    ncols = 3 if has_loss else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharex=True)
    ax_sr, ax_turns = axes[0], axes[1]

    ax_sr.set_xlabel('Episode')
    ax_sr.set_ylabel('Success Rate')
    ax_sr.set_title('Success rate')
    ax_sr.grid(True)
    ax_sr.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)

    ax_turns.set_xlabel('Episode')
    ax_turns.set_ylabel('Average turns')
    ax_turns.set_title('Average turns')
    ax_turns.grid(True)
    ax_turns.plot(numbers['x'], numbers['ave_turns'], 'b', lw=1)

    if has_loss:
        ax_bl = axes[2]
        ax_bl.set_xlabel('Episode')
        ax_bl.set_ylabel('MSE loss')
        ax_bl.set_title('DQN Bellman error (TD target MSE)')
        ax_bl.grid(True)
        ax_bl.plot(numbers['x'], numbers['bellman_loss'], 'g', lw=1)

    fig.suptitle('Learning curve')
    fig.tight_layout()
    plt.show()


def load_eval_turns_for_episode(path, episode):
    """
    Load per-dialogue turn counts from eval_dialog_turns in the JSON.
    episode: RL episode index; use -1 for the latest checkpoint in the file.
    Returns (turns_list, resolved_episode) or (None, None) if missing.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    et = data.get('eval_dialog_turns')
    if not et:
        return None, None
    keys = sorted(int(k) for k in et.keys())
    if not keys:
        return None, None
    if episode < 0:
        episode = keys[-1]
    sk = str(episode)
    if sk not in et:
        return None, episode
    turns = [int(t) for t in et[sk]]
    return turns, episode


def draw_turn_distribution(turns, episode):
    """Histogram of dialogue lengths (evaluation batch)."""
    
    fig, ax_hist = plt.subplots(figsize=(7, 4))

    # Histogram
    counts, bins, patches = ax_hist.hist(
        turns,
        bins='auto',
        color='steelblue',
        edgecolor='black',
        alpha=0.85
    )

    # Hiển thị count trên từng cột
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()

            ax_hist.text(
                x,
                y,
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    ax_hist.set_xlabel('Dialogue length (turns)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(
        'Turn distribution (RL episode %s, n=%d eval dialogs)'
        % (episode, len(turns))
    )

    ax_hist.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    plt.show()


def main(params):
    cmd = params['cmd']
    
    if cmd == 0:
        numbers = load_performance_file(params['result_file'])
        draw_learning_curve(numbers)
    elif cmd == 1:
        read_performance_records(params['result_file'])
    elif cmd == 2:
        turns, ep = load_eval_turns_for_episode(
            params['result_file'], params['turn_dist_episode'])
        if turns is None:
            if ep is not None:
                print('No turn list for RL episode %s in this file.' % ep)
            else:
                print(
                    'No eval_dialog_turns in JSON. Re-run training with an updated run_camrest.py '
                    'so simulation_epoch saves per-dialogue lengths, then try again.'
                )
            return
        draw_turn_distribution(turns, ep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cmd', dest='cmd', type=int, default=1,
                        help='0=learning curves, 1=print table, 2=turn histogram')
    
    parser.add_argument('--result_file', dest='result_file', type=str, default='./deep_dialog/checkpoints/rl_agent/11142016/noe2e/agt_9_performance_records.json', help='path to the result file')
    parser.add_argument('--turn_dist_episode', dest='turn_dist_episode', type=int, default=-1,
                        help='for cmd=2: RL episode index to plot (-1 = latest in file)')
    
    args = parser.parse_args()
    params = vars(args)
    print (json.dumps(params, indent=2))

    main(params)