"""
Train Deep Dyna-Q on the Frames hotel-booking dataset.

Usage:
    python run_frames.py                       # DQN k=5 (Dyna-Q)
    python run_frames.py --planning_steps 0    # k=0 pure DQN baseline
    python run_frames.py --episodes 500        # more training
"""

import argparse, copy, json, os, pickle, random
from datetime import datetime

import numpy
import numpy as np
import torch

# ── Patch dialog_config BEFORE any pipeline imports ──────────────────────────
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

os.chdir(os.path.dirname(os.path.abspath(__file__)) or './')

from deep_dialog import dialog_config
from deep_dialog.agents import (
    AgentDQN, InformAgent, RandomAgent, RequestAllAgent, RequestBasicsAgent,
)
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, FramesRuleSimulator


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def main():
    parser = argparse.ArgumentParser()

    data_dir = './deep_dialog/data/frames'
    parser.add_argument('--dict_path',       default=os.path.join(data_dir, 'frames_dict.p'))
    parser.add_argument('--kb_path',         default=os.path.join(data_dir, 'frames_kb.p'))
    parser.add_argument('--act_set',         default=os.path.join(data_dir, 'dia_acts_frames.txt'))
    parser.add_argument('--slot_set',        default=os.path.join(data_dir, 'slot_set_frames.txt'))
    parser.add_argument('--goal_file_path',  default=os.path.join(data_dir, 'frames_user_goals.p'))
    parser.add_argument('--diaact_nl_pairs', default=os.path.join(data_dir, 'dia_act_nl_pairs_frames.json'))

    parser.add_argument('--max_turn',        type=int,   default=20)
    parser.add_argument('--episodes',        type=int,   default=300)
    parser.add_argument('--slot_err_prob',   type=float, default=0.0)
    parser.add_argument('--slot_err_mode',   type=int,   default=0)
    parser.add_argument('--intent_err_prob', type=float, default=0.0)

    parser.add_argument('--agt',             type=int,   default=9)
    parser.add_argument('--epsilon',         type=float, default=0.5)
    parser.add_argument('--act_level',       type=int,   default=0)
    parser.add_argument('--run_mode',        type=int,   default=1)
    parser.add_argument('--auto_suggest',    type=int,   default=0)
    parser.add_argument('--cmd_input_mode',  type=int,   default=0)

    parser.add_argument('--experience_replay_pool_size', type=int,   default=10000)
    parser.add_argument('--dqn_hidden_size',             type=int,   default=256)
    parser.add_argument('--batch_size',                  type=int,   default=64)
    parser.add_argument('--gamma',                       type=float, default=0.99)
    parser.add_argument('--predict_mode',                type=bool,  default=False)
    parser.add_argument('--simulation_epoch_size',       type=int,   default=100)
    parser.add_argument('--warm_start',                  type=int,   default=1)
    parser.add_argument('--warm_start_epochs',           type=int,   default=100)
    parser.add_argument('--planning_steps',              type=int,   default=5)
    parser.add_argument('--world_model_weight',          type=float, default=0.5)
    parser.add_argument('--per_alpha',                   type=float, default=0.6)
    parser.add_argument('--per_beta',                    type=float, default=0.4)
    parser.add_argument('--target_tau',                  type=float, default=0.005)
    parser.add_argument('--learning_rate',               type=float, default=1e-3)
    parser.add_argument('--min_epsilon',                 type=float, default=0.05)
    parser.add_argument('--epsilon_decay',               type=float, default=0.992)
    parser.add_argument('--trained_model_path',          type=str,   default=None)
    parser.add_argument('-o', '--write_model_dir',       type=str,   default='./deep_dialog/checkpoints/frames/')
    parser.add_argument('--save_check_point',            type=int,   default=25)
    parser.add_argument('--success_rate_threshold',      type=float, default=0.4)
    parser.add_argument('--split_fold',                  type=int,   default=5)
    parser.add_argument('--learning_phase',              type=str,   default='all')
    parser.add_argument('--boosted',                     type=int,   default=1)
    parser.add_argument('--train_world_model',           type=int,   default=1)
    parser.add_argument('--seed',                        type=int,   default=42)
    parser.add_argument('--torch_seed',                  type=int,   default=100)
    parser.add_argument('--sample_dialog_seed',          type=int,   default=424242)
    parser.add_argument('--eval_baseline_episodes',      type=int,   default=100)
    parser.add_argument('--nlg_model_path',              type=str,   default='')
    parser.add_argument('--nlu_model_path',              type=str,   default='')
    parser.add_argument('--tensorboard',                 type=int,   default=0)
    parser.add_argument('--tensorboard_dir',             type=str,   default='')

    args   = parser.parse_args()
    params = vars(args)
    print('Dialog Parameters:')
    print(json.dumps(params, indent=2))

    numpy.random.seed(params['seed'])
    random.seed(params['seed'])
    torch.manual_seed(params['torch_seed'])

    max_turn     = params['max_turn']
    num_episodes = params['episodes']

    # ── Load data ─────────────────────────────────────────────────────────────
    all_goal_set    = load_pickle(params['goal_file_path'])
    kb              = load_pickle(params['kb_path'])
    slot_dictionary = load_pickle(params['dict_path'])
    act_set         = text_to_dict(params['act_set'])
    slot_set        = text_to_dict(params['slot_set'])

    split_fold = params['split_fold']
    goal_set   = {'train': [], 'valid': [], 'test': [], 'all': []}
    for i, g in enumerate(all_goal_set):
        (goal_set['test'] if i % split_fold == 1 else goal_set['train']).append(g)
        goal_set['all'].append(g)

    dialog_config.run_mode     = params['run_mode']
    dialog_config.auto_suggest = params['auto_suggest']

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_params = {
        'max_turn':                    max_turn,
        'epsilon':                     params['epsilon'],
        'agent_run_mode':              params['run_mode'],
        'agent_act_level':             params['act_level'],
        'experience_replay_pool_size': params['experience_replay_pool_size'],
        'dqn_hidden_size':             params['dqn_hidden_size'],
        'batch_size':                  params['batch_size'],
        'gamma':                       params['gamma'],
        'predict_mode':                params['predict_mode'],
        'trained_model_path':          params['trained_model_path'],
        'warm_start':                  params['warm_start'],
        'cmd_input_mode':              params['cmd_input_mode'],
        'world_model_weight':          params['world_model_weight'],
        'per_alpha':                   params['per_alpha'],
        'per_beta':                    params['per_beta'],
        'target_tau':                  params['target_tau'],
        'learning_rate':               params['learning_rate'],
        'min_epsilon':                 params['min_epsilon'],
        'epsilon_decay':               params['epsilon_decay'],
    }

    agt = params['agt']
    if   agt == 9:  agent = AgentDQN(kb, act_set, slot_set, agent_params)
    elif agt == 1:  agent = InformAgent(kb, act_set, slot_set, agent_params)
    elif agt == 3:  agent = RandomAgent(kb, act_set, slot_set, agent_params)
    else: raise ValueError(f"Unknown --agt {agt}")

    # ── User simulators ───────────────────────────────────────────────────────
    usersim_params = {
        'max_turn':               max_turn,
        'slot_err_probability':   params['slot_err_prob'],
        'slot_err_mode':          params['slot_err_mode'],
        'intent_err_probability': params['intent_err_prob'],
        'simulator_run_mode':     params['run_mode'],
        'simulator_act_level':    params['act_level'],
        'learning_phase':         params['learning_phase'],
        'hidden_size':            params['dqn_hidden_size'],
        'experience_replay_pool_size': params['experience_replay_pool_size'],
    }

    user_sim    = FramesRuleSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)

    # ── NLG / NLU ─────────────────────────────────────────────────────────────
    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(params['diaact_nl_pairs'])
    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)
    world_model.set_nlg_model(nlg_model)

    nlu_model = nlu()
    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)
    world_model.set_nlu_model(nlu_model)

    # ── Dialog Manager ────────────────────────────────────────────────────────
    dialog_manager = DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)

    write_model_dir = params['write_model_dir']
    os.makedirs(write_model_dir, exist_ok=True)

    batch_size        = params['batch_size']
    warm_start        = params['warm_start']
    warm_start_epochs = params['warm_start_epochs']
    planning_steps    = params['planning_steps']
    agent.planning_steps = planning_steps
    save_check_point  = params['save_check_point']

    performance_records = {
        'success_rate': {}, 'ave_turns': {}, 'ave_reward': {},
        'bellman_loss': {}, 'eval_dialog_turns': {},
    }
    best_res   = {'success_rate': 0, 'ave_reward': float('-inf'),
                  'ave_turns': float('inf'), 'epoch': 0}
    best_model = {'model': copy.deepcopy(agent)}

    _sds = int(params['sample_dialog_seed'])
    sample_dialog_rng_seed = _sds if _sds >= 0 else random.randint(0, 2**31 - 1)

    def rollout_predict_transcript(policy_agent, rng_seed):
        prev_agent = dialog_manager.agent
        try:
            numpy.random.seed(rng_seed)
            random.seed(rng_seed)
            torch.manual_seed(params['torch_seed'])
            dialog_manager.agent = policy_agent
            policy_agent.predict_mode = True
            dialog_manager.initialize_episode(use_environment=True)
            goal = copy.deepcopy(dialog_manager.user.goal)
            ep_over = False
            episode_reward = 0.0
            while not ep_over:
                ep_over, r = dialog_manager.next_turn(
                    record_training_data=False,
                    record_training_data_for_user=False)
                episode_reward += r
            return {
                'goal': goal,
                'turns': copy.deepcopy(
                    dialog_manager.state_tracker.dialog_history_dictionaries()),
                'outcome': {
                    'reward': episode_reward,
                    'turn_count': int(dialog_manager.state_tracker.turn_count),
                    'success': episode_reward > 0,
                },
            }
        finally:
            dialog_manager.agent = prev_agent
            policy_agent.predict_mode = False

    def simulation_epoch(n):
        prev_agent  = dialog_manager.agent
        prev_predict = agent.predict_mode
        try:
            dialog_manager.agent = agent
            agent.predict_mode   = True
            successes = cum_rew = cum_turns = 0
            for _ in range(n):
                dialog_manager.initialize_episode(use_environment=True)
                ep_over = False
                while not ep_over:
                    ep_over, r = dialog_manager.next_turn(record_training_data_for_user=False)
                    cum_rew += r
                    if ep_over:
                        if r > 0:
                            successes += 1
                        cum_turns += dialog_manager.state_tracker.turn_count
            return {
                'success_rate': successes / n,
                'ave_reward':   cum_rew   / n,
                'ave_turns':    cum_turns / n,
            }
        finally:
            dialog_manager.agent = prev_agent
            agent.predict_mode   = prev_predict

    def save_model(ag, epoch, best_epoch=None):
        sr   = best_res['success_rate']
        best = best_epoch or epoch
        fname = f'agt_9_{best}_{epoch}_{sr:.5f}.pkl'
        fpath = os.path.join(write_model_dir, fname)
        ag.save(fpath)
        print(f'Saved model: {fpath}')

    def save_records():
        ppath = os.path.join(write_model_dir, 'agt_9_performance_records.json')
        with open(ppath, 'w', encoding='utf-8') as f:
            json.dump(performance_records, f, indent=2)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = None
    if params['tensorboard'] and SummaryWriter:
        tb_dir = params['tensorboard_dir'] or os.path.join(write_model_dir, 'tensorboard')
        writer = SummaryWriter(tb_dir)

    # ── Warm start ───────────────────────────────────────────────────────────
    if agt == 9 and warm_start == 1:
        print(f'\n[Warm Start] {warm_start_epochs} episodes …')
        ws_suc = ws_rew = ws_turns = 0
        for _ in range(warm_start_epochs):
            dialog_manager.initialize_episode(use_environment=True)
            ep_over = False
            while not ep_over:
                ep_over, r = dialog_manager.next_turn()
                ws_rew += r
                if ep_over:
                    if r > 0:
                        ws_suc += 1
                    ws_turns += dialog_manager.state_tracker.turn_count
        if params['boosted']:
            world_model.train(batch_size, num_batches=5)
        agent.warm_start = 2
        n = max(1, warm_start_epochs)
        print(f'[Warm Start] SR={ws_suc/n:.3f}  rew={ws_rew/n:.2f}  turns={ws_turns/n:.1f}')

    # ── Baseline eval (before RL) ─────────────────────────────────────────────
    n_eval_ab = params['eval_baseline_episodes']
    metrics_ab = {}
    sample_before = None
    if agt == 9 and n_eval_ab > 0:
        print(f'[Baseline] evaluating {n_eval_ab} episodes …')
        before_sim   = simulation_epoch(n_eval_ab)
        metrics_ab   = {'eval_episodes': n_eval_ab, 'before_rl': before_sim}
        sample_before = rollout_predict_transcript(copy.deepcopy(agent), sample_dialog_rng_seed)
        print(f'[Baseline] SR={before_sim["success_rate"]:.3f}  rew={before_sim["ave_reward"]:.2f}')

    # ── Main training loop ───────────────────────────────────────────────────
    print('\n[Training]')
    sim_eps = planning_steps + 1

    for episode in range(1, num_episodes + 1):
        # One real episode
        agent.predict_mode = False
        dialog_manager.initialize_episode(use_environment=True)
        ep_over = False
        ep_reward = 0.0
        while not ep_over:
            ep_over, r = dialog_manager.next_turn(record_training_data_for_user=False)
            ep_reward += r

        if agt == 9:
            # Dyna-Q: k simulation episodes with world model
            agent.predict_mode   = True
            world_model.predict_mode = True
            for _ in range(sim_eps):
                use_env = (_ == 0)
                dialog_manager.initialize_episode(use_environment=use_env)
                ep2 = False
                while not ep2:
                    ep2, _ = dialog_manager.next_turn()
            agent.predict_mode   = False
            world_model.predict_mode = False

            # Evaluate current policy
            sim_res = simulation_epoch(params['simulation_epoch_size'])

            # Train DQN once per episode
            agent.train(batch_size, 1)
            bellman_loss = getattr(agent, 'last_avg_bellman_loss', 0.0)
            if params['train_world_model']:
                world_model.train(batch_size=batch_size, num_batches=1)

            sr        = sim_res['success_rate']
            ave_rew   = sim_res['ave_reward']
            ave_turns = sim_res['ave_turns']

            performance_records['success_rate'][str(episode)] = sr
            performance_records['ave_reward'][str(episode)]   = ave_rew
            performance_records['ave_turns'][str(episode)]    = ave_turns
            performance_records['bellman_loss'][str(episode)] = bellman_loss

            if writer:
                writer.add_scalar('SR/train', sr, episode)
                writer.add_scalar('Reward/train', ave_rew, episode)
                writer.add_scalar('Turns/train', ave_turns, episode)
                writer.add_scalar('Loss/bellman', bellman_loss, episode)

            print(f'Episode {episode:4d} | SR={sr:.3f} | rew={ave_rew:.2f} | '
                  f'turns={ave_turns:.1f} | loss={bellman_loss:.4f} | eps={agent.epsilon:.4f}')

            if sr >= best_res['success_rate']:
                best_res  = {'success_rate': sr, 'ave_reward': ave_rew,
                             'ave_turns': ave_turns, 'epoch': episode}
                best_model['model'] = copy.deepcopy(agent)

            if episode % save_check_point == 0:
                save_model(agent, episode, best_res['epoch'])
                save_records()

    # ── Final save ───────────────────────────────────────────────────────────
    save_model(best_model['model'], num_episodes, best_res['epoch'])
    save_records()

    sample_after = rollout_predict_transcript(
        copy.deepcopy(best_model['model']), sample_dialog_rng_seed)

    ppath = os.path.join(write_model_dir, 'agt_9_performance_records.json')
    with open(ppath, encoding='utf-8') as f:
        rec = json.load(f)
    if metrics_ab:
        metrics_ab['after_rl'] = best_res
        rec['metrics_ab'] = metrics_ab
    if sample_before:
        rec['sample_dialog_before'] = sample_before
    rec['sample_dialog_after'] = sample_after
    with open(ppath, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)

    print(f'\n[Done] Best SR={best_res["success_rate"]:.3f} at episode {best_res["epoch"]}')
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
