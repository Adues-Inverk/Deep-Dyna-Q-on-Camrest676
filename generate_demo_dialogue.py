"""
Generate structured dialogue JSON files for the web dashboard.

Runs the best k=0 and k=5 CamRest agents on the same user goals and
saves their conversation histories so the dashboard can display them.

Output:
  img/k0_vs_k5_dialogue.json   – comparison mode (same goal, both agents)
  img/k0_cases.json            – k=0 success vs fail case
"""
import os, sys, json, random, copy, contextlib, io
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)) or './')
sys.path.insert(0, '.')

from deep_dialog import dialog_config
from deep_dialog.agents import AgentDQN
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, RuleSimulator

import pickle

DATA_DIR = './deep_dialog/data/camrest676'
CKPT_K0  = './deep_dialog/checkpoints/best/k0'
CKPT_K5  = './deep_dialog/checkpoints/best/k5'
IMG_DIR  = './img'
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 424242

def load_p(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def find_best(ckpt_dir):
    best = (0.0, None)
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pkl') and f.startswith('agt_9') and 'performance' not in f:
            try:
                sr = float(f.replace('.pkl','').split('_')[-1])
                if sr > best[0]:
                    best = (sr, os.path.join(ckpt_dir, f))
            except ValueError:
                pass
    return best[1]

def make_agent(ckpt_path, kb, act_set, slot_set, k):
    dialog_config.run_mode = 0
    params = dict(
        max_turn=20, epsilon=0.0, agent_run_mode=0, agent_act_level=0,
        experience_replay_pool_size=100, dqn_hidden_size=256, batch_size=64,
        gamma=0.99, predict_mode=True, trained_model_path=ckpt_path,
        warm_start=0, cmd_input_mode=0, world_model_weight=0.5,
        per_alpha=0.6, per_beta=0.4, target_tau=0.005, learning_rate=1e-3,
        min_epsilon=0.0, epsilon_decay=1.0,
    )
    agent = AgentDQN(kb, act_set, slot_set, params)
    agent.predict_mode = True
    agent.planning_steps = k
    return agent

def make_dm(agent, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model):
    usersim_params = dict(
        max_turn=20, slot_err_probability=0.0, slot_err_mode=0,
        intent_err_probability=0.0, simulator_run_mode=0, simulator_act_level=0,
        learning_phase='all', hidden_size=256,
    )
    user_sim    = RuleSimulator(slot_dict, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dict, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)
    for obj in (agent, user_sim, world_model):
        obj.set_nlg_model(nlg_model)
        obj.set_nlu_model(nlu_model)
    return DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)

def run_episode(dm, seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        dm.initialize_episode(use_environment=True)
    goal = copy.deepcopy(dm.user.goal)
    over = False; reward = 0.0
    while not over:
        with contextlib.redirect_stdout(io.StringIO()):
            over, r = dm.next_turn(record_training_data=False, record_training_data_for_user=False)
        reward += r
    raw_hist = dm.state_tracker.dialog_history_dictionaries()
    turns = dm.state_tracker.turn_count
    return goal, reward, int(turns), raw_hist

def clean_history(raw_hist):
    """Convert raw history dicts to clean speaker-tagged turn list."""
    out = []
    for i, h in enumerate(raw_hist):
        speaker = 'user' if i % 2 == 0 else 'agent'
        turn = {
            'turn': i // 2 + 1,
            'speaker': speaker,
            'diaact': h.get('diaact', ''),
            'inform_slots': h.get('inform_slots', {}),
            'request_slots': h.get('request_slots', {}),
            'nl': h.get('nl', {}).get('usr' if speaker == 'user' else 'sys', ''),
        }
        out.append(turn)
    return out

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    all_goals   = load_p(os.path.join(DATA_DIR, 'camrest_user_goals.p'))
    kb          = load_p(os.path.join(DATA_DIR, 'camrest_kb.p'))
    slot_dict   = load_p(os.path.join(DATA_DIR, 'camrest_dict.p'))
    act_set     = text_to_dict(os.path.join(DATA_DIR, 'dia_acts_camrest.txt'))
    slot_set    = text_to_dict(os.path.join(DATA_DIR, 'slot_set_camrest.txt'))
    goal_set    = {'train': [], 'valid': [], 'test': [], 'all': []}
    for i, g in enumerate(all_goals):
        (goal_set['test'] if i % 5 == 1 else goal_set['train']).append(g)
        goal_set['all'].append(g)

    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(os.path.join(DATA_DIR, 'dia_act_nl_pairs_camrest.json'))
    nlu_model = nlu()

    ckpt_k0 = find_best(CKPT_K0)
    ckpt_k5 = find_best(CKPT_K5)
    print(f'k=0 checkpoint: {os.path.basename(ckpt_k0)}')
    print(f'k=5 checkpoint: {os.path.basename(ckpt_k5)}')

    agent_k0 = make_agent(ckpt_k0, kb, act_set, slot_set, k=0)
    agent_k5 = make_agent(ckpt_k5, kb, act_set, slot_set, k=5)

    dm_k0 = make_dm(agent_k0, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model)
    dm_k5 = make_dm(agent_k5, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model)

    # ── Comparison dialogue: run 8 goals, pick top candidates ─────────────────
    print('\nGenerating comparison dialogues...')
    candidates = []
    seeds = [SEED + i * 17 for i in range(15)]
    for rank, seed in enumerate(seeds, 1):
        goal_k0, rew_k0, turns_k0, hist_k0 = run_episode(dm_k0, seed)
        goal_k5, rew_k5, turns_k5, hist_k5 = run_episode(dm_k5, seed)
        candidates.append({
            'rank': rank,
            'seed': seed,
            'k0_reward': float(rew_k0),
            'k0_turns': turns_k0,
            'k5_reward': float(rew_k5),
            'k5_turns': turns_k5,
            'goal': {
                'inform_slots': goal_k0.get('inform_slots', {}),
                'request_slots': goal_k0.get('request_slots', {}),
            },
            # aliases for app.js compatibility with the reference implementation
            'ddq_reward': float(rew_k0),
            'ddq_turns': turns_k0,
            'd3q_reward': float(rew_k5),
            'd3q_turns': turns_k5,
            'ddq_history': clean_history(hist_k0),
            'd3q_history': clean_history(hist_k5),
        })
        status = f"seed={seed}: k0={rew_k0:+.0f}/{turns_k0}t  k5={rew_k5:+.0f}/{turns_k5}t"
        print(f'  [{rank:2d}] {status}')

    comparison = {'candidates': candidates, 'candidate_pool_size': len(candidates)}
    out_path = os.path.join(IMG_DIR, 'k0_vs_k5_dialogue.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f'\nSaved {out_path}')

    # ── k=0 success vs fail cases ─────────────────────────────────────────────
    print('\nFinding k=0 success / fail cases...')
    success_case = fail_case = None
    for seed in [SEED + i * 7 for i in range(30)]:
        goal, rew, turns, hist = run_episode(dm_k0, seed)
        case = {
            'seed': seed,
            'reward': float(rew),
            'turns': turns,
            'goal': {'inform_slots': goal.get('inform_slots', {}), 'request_slots': goal.get('request_slots', {})},
            'history': clean_history(hist),
        }
        if rew > 0 and success_case is None:
            success_case = case
            print(f'  Success: seed={seed}, reward={rew:+.0f}, turns={turns}')
        elif rew <= 0 and fail_case is None:
            fail_case = case
            print(f'  Fail:    seed={seed}, reward={rew:+.0f}, turns={turns}')
        if success_case and fail_case:
            break

    k0_cases = {
        'candidate_pool_size': 30,
        'success_count': 1,
        'fail_count': 1,
        'success_case': success_case,
        'fail_case': fail_case,
    }
    out_path2 = os.path.join(IMG_DIR, 'k0_cases.json')
    with open(out_path2, 'w', encoding='utf-8') as f:
        json.dump(k0_cases, f, ensure_ascii=False, indent=2)
    print(f'Saved {out_path2}')
    print('\nDone.')

if __name__ == '__main__':
    main()
