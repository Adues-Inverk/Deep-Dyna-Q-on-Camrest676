"""
Evaluate a trained Frames checkpoint on the held-out test split.

Usage:
    python eval_frames.py                         # auto-detect best checkpoint
    python eval_frames.py --model path/to/agt.pkl
    python eval_frames.py --episodes 200          # more eval dialogs
"""
import argparse, os, pickle, random, copy
import numpy, numpy as np
import torch

# ── Patch dialog_config before any pipeline imports ──────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

os.chdir(os.path.dirname(os.path.abspath(__file__)) or './')

from deep_dialog import dialog_config
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, FramesRuleSimulator


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def find_best_checkpoint(search_dir):
    """Return the .pkl with the highest SR encoded in its filename."""
    pkls = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.pkl') and f.startswith('agt_9'):
                try:
                    sr = float(f.replace('.pkl', '').split('_')[-1])
                    pkls.append((sr, os.path.join(root, f)))
                except ValueError:
                    pass
    if not pkls:
        return None
    return max(pkls, key=lambda x: x[0])[1]


def main():
    parser = argparse.ArgumentParser()
    data_dir = './deep_dialog/data/frames'
    parser.add_argument('--model',           type=str, default=None)
    parser.add_argument('--checkpoint_dir',  type=str, default='./deep_dialog/checkpoints/frames/')
    parser.add_argument('--dict_path',       default=os.path.join(data_dir, 'frames_dict.p'))
    parser.add_argument('--kb_path',         default=os.path.join(data_dir, 'frames_kb.p'))
    parser.add_argument('--act_set',         default=os.path.join(data_dir, 'dia_acts_frames.txt'))
    parser.add_argument('--slot_set',        default=os.path.join(data_dir, 'slot_set_frames.txt'))
    parser.add_argument('--goal_file_path',  default=os.path.join(data_dir, 'frames_user_goals.p'))
    parser.add_argument('--diaact_nl_pairs', default=os.path.join(data_dir, 'dia_act_nl_pairs_frames.json'))
    parser.add_argument('--episodes',        type=int, default=200)
    parser.add_argument('--split_fold',      type=int, default=5)
    parser.add_argument('--max_turn',        type=int, default=20)
    parser.add_argument('--seed',            type=int, default=99)
    args = parser.parse_args()

    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Find checkpoint ───────────────────────────────────────────────────────
    model_path = args.model or find_best_checkpoint(args.checkpoint_dir)
    if model_path is None:
        print(f'No checkpoint found in {args.checkpoint_dir}')
        print('Run training first:  python run_frames.py --episodes 300')
        return
    print(f'Loading checkpoint: {model_path}')
    with open(model_path, 'rb') as f:
        agent = pickle.load(f, encoding='latin1')
    agent.predict_mode = True

    # ── Load data ─────────────────────────────────────────────────────────────
    all_goals       = load_pickle(args.goal_file_path)
    kb              = load_pickle(args.kb_path)
    slot_dictionary = load_pickle(args.dict_path)
    act_set         = text_to_dict(args.act_set)
    slot_set        = text_to_dict(args.slot_set)

    # Build test split (same split_fold as training)
    test_goals = [g for i, g in enumerate(all_goals) if i % args.split_fold == 1]
    goal_set   = {'train': [], 'valid': [], 'test': test_goals, 'all': all_goals}
    print(f'Test goals: {len(test_goals)}  (split_fold={args.split_fold})')

    dialog_config.run_mode = 1

    usersim_params = {
        'max_turn': args.max_turn,
        'slot_err_probability': 0.0,
        'slot_err_mode': 0,
        'intent_err_probability': 0.0,
        'simulator_run_mode': 1,
        'simulator_act_level': 0,
        'learning_phase': 'test',          # ← test goals only
        'hidden_size': 256,
        'experience_replay_pool_size': 1000,
    }

    user_sim    = FramesRuleSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)

    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(args.diaact_nl_pairs)
    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)
    world_model.set_nlg_model(nlg_model)

    nlu_model = nlu()
    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)
    world_model.set_nlu_model(nlu_model)

    dialog_manager = DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    successes = 0
    total_reward = 0.0
    total_turns  = 0

    print(f'\nEvaluating {args.episodes} test episodes …')
    for ep in range(args.episodes):
        dialog_manager.initialize_episode(use_environment=True)
        ep_over = False
        ep_reward = 0.0
        while not ep_over:
            ep_over, r = dialog_manager.next_turn(
                record_training_data=False,
                record_training_data_for_user=False)
            ep_reward += r
        total_reward += ep_reward
        total_turns  += dialog_manager.state_tracker.turn_count
        if ep_reward > 0:
            successes += 1

    sr        = successes / args.episodes
    ave_rew   = total_reward / args.episodes
    ave_turns = total_turns  / args.episodes

    print(f'\n{"="*45}')
    print(f'  Test Success Rate : {sr:.3f}  ({successes}/{args.episodes})')
    print(f'  Avg Reward        : {ave_rew:.2f}')
    print(f'  Avg Turns         : {ave_turns:.1f}')
    print(f'  Checkpoint        : {os.path.basename(model_path)}')
    print(f'{"="*45}')


if __name__ == '__main__':
    main()
