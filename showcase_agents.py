#!/usr/bin/env python3
"""
Showcase: React (Reason+Act) vs MuZero Agents on CamRest676

This script trains and evaluates both new agents and compares their performance
with baseline agents on the CamRest676 restaurant booking domain.

Usage:
    python showcase_agents.py --episodes 30 --agents react muzero dqn --save-report
"""

import argparse
import copy
import json
import os
import pickle
import random
from datetime import datetime
from collections import defaultdict

import numpy
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)) or './')

from deep_dialog import dialog_config
from deep_dialog.agents import (
    AgentReact,
    AgentMuZero,
    AgentDQN,
    RequestBasicsAgent,
)
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, RuleSimulator


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def evaluate_agent(agent, dialog_manager, num_episodes, agent_name="Agent"):
    """Evaluate agent performance over multiple episodes"""
    metrics = {
        'successes': 0,
        'avg_reward': 0,
        'avg_turns': 0,
        'total_reward': 0,
        'total_turns': 0,
        'dialog_lengths': [],
        'dialog_rewards': [],
        'sample_dialogs': [],
    }
    
    for episode in range(num_episodes):
        agent.predict_mode = True
        dialog_manager.agent = agent
        dialog_manager.initialize_episode(use_environment=True)
        
        episode_over = False
        episode_reward = 0
        turn_count = 0
        dialog_history = []
        
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn(
                record_training_data=False,
                record_training_data_for_user=False,
            )
            episode_reward += reward
            turn_count = int(dialog_manager.state_tracker.turn_count)
            dialog_history.append({
                'turn': turn_count,
                'reward': reward,
            })
        
        metrics['total_reward'] += episode_reward
        metrics['total_turns'] += turn_count
        metrics['dialog_lengths'].append(turn_count)
        metrics['dialog_rewards'].append(episode_reward)
        
        if episode_reward > 0:
            metrics['successes'] += 1
        
        # Store first few dialogs as samples
        if episode < 3:
            metrics['sample_dialogs'].append({
                'episode': episode,
                'success': episode_reward > 0,
                'reward': episode_reward,
                'turns': turn_count,
                'history': dialog_history[:5],  # First 5 turns
            })
        
        if (episode + 1) % 10 == 0:
            print(f"  {agent_name} - Episode {episode + 1}/{num_episodes}")
    
    metrics['avg_reward'] = metrics['total_reward'] / num_episodes
    metrics['avg_turns'] = metrics['total_turns'] / num_episodes
    metrics['success_rate'] = metrics['successes'] / num_episodes
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Showcase React and MuZero agents')
    
    data_dir = './deep_dialog/data/camrest676'
    parser.add_argument('--dict_path', default=os.path.join(data_dir, 'camrest_dict.p'))
    parser.add_argument('--kb_path', default=os.path.join(data_dir, 'camrest_kb.p'))
    parser.add_argument('--act_set', default=os.path.join(data_dir, 'dia_acts_camrest.txt'))
    parser.add_argument('--slot_set', default=os.path.join(data_dir, 'slot_set_camrest.txt'))
    parser.add_argument('--goal_file_path', default=os.path.join(data_dir, 'camrest_user_goals.p'))
    parser.add_argument('--diaact_nl_pairs', default=os.path.join(data_dir, 'dia_act_nl_pairs_camrest.json'))
    
    parser.add_argument('--max_turn', type=int, default=20)
    parser.add_argument('--episodes', type=int, default=30,
                        help='Training episodes for each agent')
    parser.add_argument('--eval_episodes', type=int, default=50,
                        help='Evaluation episodes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--torch_seed', type=int, default=100)
    parser.add_argument('--slot_err_prob', type=float, default=0.0)
    parser.add_argument('--agents', nargs='+', default=['react', 'muzero', 'dqn', 'baseline'],
                        help='Agents to showcase')
    parser.add_argument('--save-report', action='store_true',
                        help='Save detailed report to file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--warm_start_epochs', type=int, default=50)
    
    args = parser.parse_args()
    params = vars(args)
    
    print("="*80)
    print("SHOWCASE: React (Reason+Act) vs MuZero Agents on CamRest676")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Episodes (training): {params['episodes']}")
    print(f"  Episodes (evaluation): {params['eval_episodes']}")
    print(f"  Agents: {', '.join(params['agents'])}")
    print()
    
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])
    torch.manual_seed(params['torch_seed'])
    
    max_turn = params['max_turn']
    num_episodes = params['episodes']
    eval_episodes = params['eval_episodes']
    
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    all_goal_set = load_pickle(params['goal_file_path'])
    kb = load_pickle(params['kb_path'])
    slot_dictionary = load_pickle(params['dict_path'])
    
    act_set = text_to_dict(params['act_set'])
    slot_set = text_to_dict(params['slot_set'])
    
    split_fold = 5
    goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
    for i, g in enumerate(all_goal_set):
        (goal_set['test'] if i % split_fold == 1 else goal_set['train']).append(g)
        goal_set['all'].append(g)
    
    dialog_config.run_mode = 1
    
    # ------------------------------------------------------------------
    # Setup agents
    # ------------------------------------------------------------------
    print("\nInitializing agents...")
    agent_list = {}
    agent_params = {
        'max_turn': max_turn,
        'epsilon': 0.1,
        'agent_run_mode': 1,
        'agent_act_level': 0,
        'experience_replay_pool_size': 5000,
        'dqn_hidden_size': 60,
        'batch_size': params['batch_size'],
        'gamma': 0.9,
        'predict_mode': False,
        'trained_model_path': None,
        'warm_start': 1,
        'cmd_input_mode': 0,
    }
    
    if 'react' in params['agents']:
        agent_list['React'] = AgentReact(kb, act_set, slot_set, agent_params)
        print("  ✓ React Agent initialized")
    
    if 'muzero' in params['agents']:
        agent_list['MuZero'] = AgentMuZero(kb, act_set, slot_set, agent_params)
        print("  ✓ MuZero Agent initialized")
    
    if 'dqn' in params['agents']:
        agent_list['DQN'] = AgentDQN(kb, act_set, slot_set, agent_params)
        print("  ✓ DQN Agent initialized")
    
    if 'baseline' in params['agents']:
        agent_list['RequestBasics'] = RequestBasicsAgent(kb, act_set, slot_set, agent_params)
        print("  ✓ RequestBasics Baseline initialized")
    
    # ------------------------------------------------------------------
    # User simulator
    # ------------------------------------------------------------------
    print("\nSetting up user simulator...")
    usersim_params = {
        'max_turn': max_turn,
        'slot_err_probability': params['slot_err_prob'],
        'slot_err_mode': 0,
        'intent_err_probability': 0.0,
        'simulator_run_mode': 1,
        'simulator_act_level': 0,
        'learning_phase': 'all',
        'hidden_size': 60,
    }
    
    user_sim = RuleSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    
    for agent_name, agent in agent_list.items():
        agent.set_user_planning(world_model)
    
    # ------------------------------------------------------------------
    # NLG/NLU setup
    # ------------------------------------------------------------------
    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(params['diaact_nl_pairs'])
    nlu_model = nlu()
    
    for agent_name, agent in agent_list.items():
        agent.set_nlg_model(nlg_model)
        agent.set_nlu_model(nlu_model)
    
    user_sim.set_nlg_model(nlg_model)
    user_sim.set_nlu_model(nlu_model)
    world_model.set_nlg_model(nlg_model)
    world_model.set_nlu_model(nlu_model)
    
    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING PHASE")
    print("="*80)
    
    results = {}
    
    for agent_name, agent in agent_list.items():
        print(f"\n{'─'*40}")
        print(f"Training {agent_name} Agent")
        print(f"{'─'*40}")
        
        dialog_manager = DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)
        
        total_reward = 0
        total_turns = 0
        successes = 0
        
        # Warm start
        if hasattr(agent, 'warm_start') and agent.warm_start == 1:
            print(f"Warm-start phase ({params['warm_start_epochs']} episodes)...")
            for episode in range(params['warm_start_epochs']):
                dialog_manager.initialize_episode(use_environment=True)
                episode_over = False
                while not episode_over:
                    episode_over, reward = dialog_manager.next_turn()
                    total_reward += reward
                    if episode_over:
                        tc = int(dialog_manager.state_tracker.turn_count)
                        total_turns += tc
                        if reward > 0:
                            successes += 1
                
                if (episode + 1) % 20 == 0:
                    current_sr = successes / (episode + 1)
                    print(f"  Warm-start episode {episode + 1}, success rate: {current_sr:.3f}")
            
            if hasattr(agent, 'warm_start'):
                agent.warm_start = 2
                print(f"  Warm-start success rate: {successes / params['warm_start_epochs']:.3f}")
        
        # Main training
        print(f"Main training phase ({num_episodes} episodes)...")
        for episode in range(num_episodes):
            agent.predict_mode = False
            dialog_manager.agent = agent
            dialog_manager.initialize_episode(use_environment=True)
            
            episode_over = False
            while not episode_over:
                episode_over, reward = dialog_manager.next_turn(record_training_data_for_user=False)
                total_reward += reward
                if episode_over:
                    tc = int(dialog_manager.state_tracker.turn_count)
                    total_turns += tc
                    if reward > 0:
                        successes += 1
            
            # Training updates
            if hasattr(agent, 'running_experience_pool'):
                agent.running_experience_pool = agent.experience_replay_pool
                if len(agent.experience_replay_pool) > params['batch_size']:
                    agent.train(params['batch_size'], 1)
                    agent.reset_dqn_target()
            
            if (episode + 1) % 10 == 0:
                current_sr = successes / (episode + 1)
                current_ar = total_reward / (episode + 1)
                print(f"  Episode {episode + 1}: Success rate={current_sr:.3f}, Avg reward={current_ar:.2f}")
        
        print(f"Training summary for {agent_name}:")
        print(f"  Success rate: {successes / (num_episodes + params['warm_start_epochs']):.3f}")
        print(f"  Avg reward: {total_reward / (num_episodes + params['warm_start_epochs']):.2f}")
        print(f"  Avg turns: {total_turns / (num_episodes + params['warm_start_epochs']):.2f}")
    
    # ------------------------------------------------------------------
    # Evaluation phase
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)
    
    dialog_manager = DialogManager(None, user_sim, world_model, act_set, slot_set, kb)
    
    for agent_name, agent in agent_list.items():
        print(f"\n{'─'*40}")
        print(f"Evaluating {agent_name} Agent ({eval_episodes} episodes)")
        print(f"{'─'*40}")
        
        metrics = evaluate_agent(agent, dialog_manager, eval_episodes, agent_name)
        results[agent_name] = metrics
        
        print(f"\nResults for {agent_name}:")
        print(f"  ✓ Success rate:  {metrics['success_rate']:.3%}")
        print(f"  ✓ Avg reward:    {metrics['avg_reward']:.2f}")
        print(f"  ✓ Avg turns:     {metrics['avg_turns']:.2f}")
        print(f"  ✓ Min reward:    {min(metrics['dialog_rewards']):.2f}")
        print(f"  ✓ Max reward:    {max(metrics['dialog_rewards']):.2f}")
    
    # ------------------------------------------------------------------
    # Comparison and reporting
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison table
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Agent", "Success Rate", "Avg Reward", "Avg Turns"))
    print("─" * 65)
    for agent_name in sorted(results.keys()):
        m = results[agent_name]
        print("{:<20} {:<15.3%} {:<15.2f} {:<15.2f}".format(
            agent_name,
            m['success_rate'],
            m['avg_reward'],
            m['avg_turns']
        ))
    
    # Determine best agent
    best_agent = max(results.items(), key=lambda x: x[1]['success_rate'])
    print(f"\n🏆 Best Agent (by success rate): {best_agent[0]} ({best_agent[1]['success_rate']:.3%})")
    
    # Save report
    if params.get('save_report'):
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': params,
            'results': {
                agent_name: {
                    'success_rate': m['success_rate'],
                    'avg_reward': m['avg_reward'],
                    'avg_turns': m['avg_turns'],
                    'dialog_rewards': m['dialog_rewards'],
                    'dialog_lengths': m['dialog_lengths'],
                    'sample_dialogs': m['sample_dialogs'],
                }
                for agent_name, m in results.items()
            }
        }
        
        report_path = './showcase_results.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Detailed report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("Showcase complete!")
    print("="*80)


if __name__ == '__main__':
    main()
