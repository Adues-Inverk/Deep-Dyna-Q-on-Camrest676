"""
Generate structured dialogue JSON files for the Frames dashboard.

Runs both k=0 and k=5 agents on the same hotel-booking goals and saves
side-by-side conversation histories for the web dashboard.

Output:
  img/frames_k0_vs_k5_dialogue.json  – comparison mode (same goal, both agents)
  img/frames_k0_cases.json           – k=0 success vs fail case

Run this AFTER both training runs complete:
  python run_frames.py --planning_steps 0 -o ./deep_dialog/checkpoints/frames_best/k0/
  python run_frames.py --planning_steps 5 -o ./deep_dialog/checkpoints/frames_best/k5/
"""
import os, sys, json, random, copy, contextlib, io
import numpy as np
import torch

# ── Patch dialog_config before any pipeline imports ───────────────────────
import deep_dialog.frames_dialog_config as _fdc
import deep_dialog.dialog_config as _dc
for _k in ['feasible_actions', 'feasible_actions_users',
           'INFORMABLE_SLOTS', 'REQUESTABLE_SLOTS',
           'sys_request_slots', 'sys_inform_slots',
           'sys_inform_slots_for_user', 'sys_request_slots_for_user',
           'start_dia_acts', 'NON_KB_SLOTS',
           'FAILED_DIALOG', 'SUCCESS_DIALOG', 'NO_OUTCOME_YET',
           'SUCCESS_REWARD', 'FAILURE_REWARD', 'PER_TURN_REWARD',
           'I_DO_NOT_CARE', 'NO_VALUE_MATCH', 'TICKET_AVAILABLE',
           'CONSTRAINT_CHECK_FAILURE', 'CONSTRAINT_CHECK_SUCCESS']:
    setattr(_dc, _k, getattr(_fdc, _k))

os.chdir(os.path.dirname(os.path.abspath(__file__)) or './')
sys.path.insert(0, '.')

from deep_dialog import dialog_config
from deep_dialog.agents import AgentDQN
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, FramesRuleSimulator
import pickle

DATA_DIR  = './deep_dialog/data/frames'
CKPT_K0   = './deep_dialog/checkpoints/frames_best/k0'
CKPT_K5   = './deep_dialog/checkpoints/frames_best/k5'
IMG_DIR   = './img'
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 424242

def load_p(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def find_best(ckpt_dir):
    candidates = []
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pkl') and f.startswith('agt_9') and 'performance' not in f:
            try:
                parts = f.replace('.pkl', '').split('_')
                sr    = float(parts[-1])
                ep    = int(parts[-2])
                candidates.append((sr, ep, os.path.join(ckpt_dir, f)))
            except (ValueError, IndexError):
                pass
    if not candidates:
        return None
    # prefer highest SR; break ties by latest episode
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]

def make_agent(ckpt_path, kb, act_set, slot_set, k):
    dialog_config.run_mode = 1
    params = dict(
        max_turn=20, epsilon=0.0, agent_run_mode=1, agent_act_level=0,
        experience_replay_pool_size=100, dqn_hidden_size=256, batch_size=64,
        gamma=0.99, predict_mode=True, trained_model_path=None,
        warm_start=0, cmd_input_mode=0, world_model_weight=0.5,
        per_alpha=0.6, per_beta=0.4, target_tau=0.005, learning_rate=1e-3,
        min_epsilon=0.0, epsilon_decay=1.0,
    )
    agent = AgentDQN(kb, act_set, slot_set, params)
    agent.load(ckpt_path)
    agent.predict_mode = True
    agent.planning_steps = k
    return agent

def make_dm(agent, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model):
    up = dict(
        max_turn=20, slot_err_probability=0.0, slot_err_mode=0,
        intent_err_probability=0.0, simulator_run_mode=1, simulator_act_level=0,
        learning_phase='test', hidden_size=256, experience_replay_pool_size=1000,
    )
    user_sim    = FramesRuleSimulator(slot_dict, act_set, slot_set, goal_set, up)
    world_model = ModelBasedSimulator(slot_dict, act_set, slot_set, goal_set, up)
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
    raw   = dm.state_tracker.dialog_history_dictionaries()
    turns = dm.state_tracker.turn_count
    return goal, reward, int(turns), raw

def clean_history(raw):
    out = []
    for i, h in enumerate(raw):
        speaker = 'user' if i % 2 == 0 else 'agent'
        out.append({
            'turn':          i // 2 + 1,
            'speaker':       speaker,
            'diaact':        h.get('diaact', ''),
            'inform_slots':  h.get('inform_slots', {}),
            'request_slots': h.get('request_slots', {}),
            'nl':            h.get('nl', {}).get('usr' if speaker == 'user' else 'sys', ''),
        })
    return out

def main():
    all_goals = load_p(os.path.join(DATA_DIR, 'frames_user_goals.p'))
    kb        = load_p(os.path.join(DATA_DIR, 'frames_kb.p'))
    slot_dict = load_p(os.path.join(DATA_DIR, 'frames_dict.p'))
    act_set   = text_to_dict(os.path.join(DATA_DIR, 'dia_acts_frames.txt'))
    slot_set  = text_to_dict(os.path.join(DATA_DIR, 'slot_set_frames.txt'))

    goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
    for i, g in enumerate(all_goals):
        (goal_set['test'] if i % 5 == 1 else goal_set['train']).append(g)
        goal_set['all'].append(g)

    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(os.path.join(DATA_DIR, 'dia_act_nl_pairs_frames.json'))
    nlu_model = nlu()

    ckpt_k0 = find_best(CKPT_K0)
    ckpt_k5 = find_best(CKPT_K5)
    print(f'k=0 checkpoint: {os.path.basename(ckpt_k0)}')
    print(f'k=5 checkpoint: {os.path.basename(ckpt_k5)}')

    agent_k0 = make_agent(ckpt_k0, kb, act_set, slot_set, k=0)
    agent_k5 = make_agent(ckpt_k5, kb, act_set, slot_set, k=5)

    dm_k0 = make_dm(agent_k0, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model)
    dm_k5 = make_dm(agent_k5, kb, slot_dict, act_set, slot_set, goal_set, nlg_model, nlu_model)

    # ── Comparison pool: 15 goals, same seed for both agents ──────────────
    print('\nGenerating comparison dialogues...')
    candidates = []
    seeds = [SEED + i * 17 for i in range(15)]
    for rank, seed in enumerate(seeds, 1):
        goal_k0, rew_k0, turns_k0, hist_k0 = run_episode(dm_k0, seed)
        goal_k5, rew_k5, turns_k5, hist_k5 = run_episode(dm_k5, seed)
        candidates.append({
            'rank':  rank,
            'seed':  seed,
            'k0_reward': float(rew_k0), 'k0_turns': turns_k0,
            'k5_reward': float(rew_k5), 'k5_turns': turns_k5,
            # aliases so the JS can use ddq_history / d3q_history too
            'ddq_reward': float(rew_k0), 'ddq_turns': turns_k0,
            'd3q_reward': float(rew_k5), 'd3q_turns': turns_k5,
            'goal': {
                'inform_slots':  goal_k0.get('inform_slots',  {}),
                'request_slots': goal_k0.get('request_slots', {}),
            },
            'k0_history':  clean_history(hist_k0),
            'k5_history':  clean_history(hist_k5),
            'ddq_history': clean_history(hist_k0),
            'd3q_history': clean_history(hist_k5),
        })
        tag_k0 = '+' if rew_k0 > 0 else ' '
        tag_k5 = '+' if rew_k5 > 0 else ' '
        print(f'  [{rank:2d}] seed={seed}: k0={tag_k0}{rew_k0:.0f}/{turns_k0}t  k5={tag_k5}{rew_k5:.0f}/{turns_k5}t')

    out1 = os.path.join(IMG_DIR, 'frames_k0_vs_k5_dialogue.json')
    with open(out1, 'w', encoding='utf-8') as f:
        json.dump({'candidates': candidates, 'candidate_pool_size': len(candidates)}, f, ensure_ascii=False, indent=2)
    print(f'\nSaved {out1}')

    # ── k=0 success vs fail cases ──────────────────────────────────────────
    print('\nFinding k=0 success / fail cases...')
    success_case = fail_case = None
    for seed in [SEED + i * 7 for i in range(50)]:
        goal, rew, turns, raw = run_episode(dm_k0, seed)
        case = {
            'seed': seed, 'reward': float(rew), 'turns': turns,
            'goal': {'inform_slots': goal.get('inform_slots', {}),
                     'request_slots': goal.get('request_slots', {})},
            'history': clean_history(raw),
        }
        if rew > 0 and success_case is None:
            success_case = case
            print(f'  Success: seed={seed}  reward={rew:+.0f}  turns={turns}  goal={goal.get("inform_slots",{})}')
        elif rew <= 0 and fail_case is None:
            fail_case = case
            print(f'  Fail:    seed={seed}  reward={rew:+.0f}  turns={turns}')
        if success_case and fail_case:
            break

    out2 = os.path.join(IMG_DIR, 'frames_k0_cases.json')
    with open(out2, 'w', encoding='utf-8') as f:
        json.dump({
            'candidate_pool_size': 50,
            'success_count': 1, 'fail_count': 1,
            'success_case': success_case,
            'fail_case':    fail_case,
        }, f, ensure_ascii=False, indent=2)
    print(f'Saved {out2}\nDone.')

if __name__ == '__main__':
    main()
