"""
React (Reason + Act) Agent for Dialog System

This agent uses a reasoning component to analyze the dialog state and generate
chain-of-thought reasoning before selecting actions. This combines explicit 
reasoning with action selection for improved interpretability.
"""

import random
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from deep_dialog import dialog_config
from ..agent import Agent

DEVICE = torch.device('cpu')
Transition = namedtuple('Transition', ('state', 'reasoning', 'action', 'reward', 'next_state', 'term'))


class ReasoningModule(nn.Module):
    """
    Reasoning module that analyzes dialog state and generates reasoning embeddings.
    This module encodes the reasoning process about dialog context.
    """
    def __init__(self, state_dim, hidden_size):
        super(ReasoningModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.reasoning_output = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        reasoning = self.reasoning_output(x)
        return reasoning


class ActionModule(nn.Module):
    """
    Action module that takes reasoning and state to produce action probabilities.
    """
    def __init__(self, state_dim, reasoning_dim, hidden_size, num_actions):
        super(ActionModule, self).__init__()
        self.fc1 = nn.Linear(state_dim + reasoning_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, reasoning):
        combined = torch.cat([state, reasoning], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class AgentReact(Agent):
    """
    React Agent: Reason + Act paradigm for dialog management
    
    Combines explicit reasoning about dialog state with action selection.
    Features:
    - Reasoning module to analyze context
    - Policy network for action selection
    - Value network for reward estimation
    - Experience replay from real and simulated trajectories
    """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        
        self.experience_replay_pool = deque(maxlen=self.experience_replay_pool_size)
        self.experience_replay_pool_from_model = deque(maxlen=self.experience_replay_pool_size)
        self.running_experience_pool = None
        
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.reasoning_dim = params.get('reasoning_dim', 40)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        # Initialize reasoning and action modules
        self.reasoning_module = ReasoningModule(self.state_dimension, self.hidden_size).to(DEVICE)
        self.action_module = ActionModule(
            self.state_dimension, 
            self.reasoning_dim, 
            self.hidden_size, 
            self.num_actions
        ).to(DEVICE)
        
        # Target networks
        self.target_reasoning_module = ReasoningModule(self.state_dimension, self.hidden_size).to(DEVICE)
        self.target_action_module = ActionModule(
            self.state_dimension,
            self.reasoning_dim,
            self.hidden_size,
            self.num_actions
        ).to(DEVICE)
        
        self.target_reasoning_module.load_state_dict(self.reasoning_module.state_dict())
        self.target_action_module.load_state_dict(self.action_module.state_dict())
        self.target_reasoning_module.eval()
        self.target_action_module.eval()
        
        # Optimizers
        self.optimizer_reasoning = optim.Adam(self.reasoning_module.parameters(), lr=1e-3)
        self.optimizer_action = optim.Adam(self.action_module.parameters(), lr=1e-3)
        
        self.cur_bellman_err = 0
        self.last_avg_bellman_loss = 0.0
        self.reasoning_history = []
        
        # Prediction Mode: load trained model
        if params.get('trained_model_path'):
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2
    
    def initialize_episode(self):
        """Initialize episode state"""
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = list(dialog_config.sys_request_slots)
        self._user_request_slots = []
        self.reasoning_history = []
    
    def prepare_state_representation(self, state):
        """Prepare state representation for reasoning and action modules"""
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']
        
        # User act representation
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        
        # User inform slots
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0
        
        # User request slots
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0
        
        # Current slots
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0
        
        # Last agent act
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        
        # Last agent inform slots
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0
        
        # Last agent request slots
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
        
        # Turn representation
        turn_rep = np.zeros((1, 1))
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0
        
        # KB results
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        
        self.final_representation = np.hstack([
            user_act_rep, user_inform_slots_rep, user_request_slots_rep,
            agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep,
            current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep
        ])
        
        return self.final_representation
    
    def state_to_action(self, state):
        """Generate reasoning and select action"""
        self._user_request_slots = [
            s for s in state['current_slots']['request_slots'].keys()
            if s != 'taskcomplete'
        ]
        
        self.representation = self.prepare_state_representation(state)
        self.action, reasoning_text = self.run_policy(self.representation)
        self.reasoning_history.append(reasoning_text)
        
        if isinstance(self.action, torch.Tensor):
            self.action = int(self.action.view(-1)[0].item())
        
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
    
    def generate_reasoning(self, state_rep):
        """Generate reasoning about current state"""
        # Count filled slots
        filled_slots = np.sum(state_rep[0, self.act_cardinality:self.act_cardinality + self.slot_cardinality])
        user_act_idx = np.argmax(state_rep[0, :self.act_cardinality])
        user_act = [k for k, v in self.act_set.items() if v == user_act_idx][0]
        
        reasoning = f"State: User act={user_act}, filled_slots={int(filled_slots)}"
        return reasoning
    
    def run_policy(self, representation):
        """Epsilon-greedy policy with reasoning"""
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            reasoning = f"Random action: {self.feasible_actions[action]['diaact']}"
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                action = self.rule_policy()
                reasoning = f"Rule-based action: {self.feasible_actions[action]['diaact']}"
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(representation)
                    reasoning_embed = self.reasoning_module(state_tensor)
                    policy, value = self.action_module(state_tensor, reasoning_embed)
                    action = torch.argmax(policy, dim=1).item()
                    reasoning = self.generate_reasoning(representation)
                    reasoning += f" -> Action: {self.feasible_actions[action]['diaact']}"
        
        return action, reasoning
    
    def rule_policy(self):
        """Rule-based policy for warm start"""
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1
            act_slot_response = {'diaact': "request", 'inform_slots': {}, 'request_slots': {slot: "UNK"}}
        elif self._user_request_slots:
            slot = self._user_request_slots[0]
            act_slot_response = {'diaact': "inform", 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}}
            self.phase += 1
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}
        
        return self.action_index(act_slot_response)
    
    def action_index(self, act_slot_response):
        """Get action index from action specification"""
        for i, action in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception(f"Action not found: {act_slot_response}")
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """Register experience for training"""
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        reasoning = self.reasoning_history[-1] if self.reasoning_history else "No reasoning"
        
        training_example = (state_t_rep, reasoning, action_t, reward_t, state_tplus1_rep, episode_over, st_user)
        
        if not self.predict_mode:
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)
    
    def sample_from_buffer(self, batch_size):
        """Sample from experience replay buffer"""
        batch = [random.choice(self.running_experience_pool) for _ in range(batch_size)]
        states = torch.FloatTensor(np.array([t[0].squeeze() for t in batch]))
        actions = torch.LongTensor([t[2] for t in batch])
        rewards = torch.FloatTensor([t[3] for t in batch])
        next_states = torch.FloatTensor(np.array([t[4].squeeze() for t in batch]))
        terminals = torch.FloatTensor([t[5] for t in batch])
        
        return states, actions, rewards, next_states, terminals
    
    def train(self, batch_size, num_epochs):
        """Train the React agent"""
        if not self.running_experience_pool or len(self.running_experience_pool) < batch_size:
            return
        
        for epoch in range(num_epochs):
            states, actions, rewards, next_states, terminals = self.sample_from_buffer(batch_size)
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            terminals = terminals.to(DEVICE)
            
            # Forward pass
            with torch.no_grad():
                next_reasoning = self.target_reasoning_module(next_states)
                _, next_values = self.target_action_module(next_states, next_reasoning)
                target_values = rewards + (1 - terminals) * self.gamma * next_values.squeeze()
            
            current_reasoning = self.reasoning_module(states)
            policy, values = self.action_module(states, current_reasoning)
            
            # Policy loss
            action_one_hot = torch.zeros(batch_size, self.num_actions).to(DEVICE)
            action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
            policy_loss = -torch.sum(action_one_hot * torch.log_softmax(policy, dim=1), dim=1).mean()
            
            # Value loss
            value_loss = torch.nn.functional.smooth_l1_loss(values.squeeze(), target_values)
            
            total_loss = policy_loss + value_loss
            
            self.optimizer_reasoning.zero_grad()
            self.optimizer_action.zero_grad()
            total_loss.backward()
            self.optimizer_reasoning.step()
            self.optimizer_action.step()
            
            self.last_avg_bellman_loss = total_loss.item()
    
    def reset_dqn_target(self):
        """Update target networks"""
        self.target_reasoning_module.load_state_dict(self.reasoning_module.state_dict())
        self.target_action_module.load_state_dict(self.action_module.state_dict())
    
    def set_user_planning(self, user_planning):
        """Set world model for planning"""
        self.world_model = user_planning
    
    def save(self, path):
        """Save model"""
        torch.save({
            'reasoning_module': self.reasoning_module.state_dict(),
            'action_module': self.action_module.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.reasoning_module.load_state_dict(checkpoint['reasoning_module'])
        self.action_module.load_state_dict(checkpoint['action_module'])
