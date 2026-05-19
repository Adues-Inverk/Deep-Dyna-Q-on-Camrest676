"""
MuZero Agent for Dialog System

Implements a simplified version of DeepMind's MuZero algorithm.
MuZero learns a world model that predicts future latent states, policies, and values
without needing to learn explicit state transitions.

Key features:
- Representation network: encodes observations into latent states
- Dynamics network: predicts next latent state from action
- Prediction network: predicts policy and value from latent state
- Monte Carlo Tree Search for planning
"""

import random
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple, defaultdict
from deep_dialog import dialog_config
from ..agent import Agent
import math

DEVICE = torch.device('cpu')
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class RepresentationNetwork(nn.Module):
    """Encodes observation into latent state representation"""
    def __init__(self, state_dim, latent_dim, hidden_size):
        super(RepresentationNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        latent_state = self.fc3(x)
        return latent_state


class DynamicsNetwork(nn.Module):
    """Predicts next latent state and immediate reward from current state and action"""
    def __init__(self, latent_dim, action_dim, hidden_size):
        super(DynamicsNetwork, self).__init__()
        # State and action encoding
        self.action_embedding = nn.Linear(action_dim, hidden_size // 2)
        self.fc1 = nn.Linear(latent_dim + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.state_out = nn.Linear(hidden_size, latent_dim)
        self.reward_out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, latent_state, action_one_hot):
        action_emb = self.relu(self.action_embedding(action_one_hot))
        x = torch.cat([latent_state, action_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        next_state = self.state_out(x)
        reward = self.reward_out(x)
        return next_state, reward


class PredictionNetwork(nn.Module):
    """Predicts policy and value from latent state"""
    def __init__(self, latent_dim, hidden_size, num_actions):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, latent_state):
        x = self.relu(self.fc1(latent_state))
        x = self.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class MCTSNode:
    """Node in Monte Carlo Tree Search tree"""
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = prior
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count, c=1.25):
        pb_c = c * math.sqrt(parent_visit_count) / (self.visit_count + 1)
        prior_score = self.prior * math.sqrt(parent_visit_count) / (self.visit_count + 1)
        return self.value() + pb_c + prior_score


class AgentMuZero(Agent):
    """
    MuZero Agent: Model-based planning with learned world model
    
    Features:
    - Learns latent state representation
    - Learns world dynamics (state transition + reward)
    - Learns value and policy functions
    - Uses Monte Carlo Tree Search for planning
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
        self.latent_dim = params.get('latent_dim', 32)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        # Initialize MuZero networks
        self.representation_network = RepresentationNetwork(
            self.state_dimension, self.latent_dim, self.hidden_size
        ).to(DEVICE)
        
        self.dynamics_network = DynamicsNetwork(
            self.latent_dim, self.num_actions, self.hidden_size
        ).to(DEVICE)
        
        self.prediction_network = PredictionNetwork(
            self.latent_dim, self.hidden_size, self.num_actions
        ).to(DEVICE)
        
        # Target networks for stability
        self.target_prediction_network = PredictionNetwork(
            self.latent_dim, self.hidden_size, self.num_actions
        ).to(DEVICE)
        self.target_prediction_network.load_state_dict(self.prediction_network.state_dict())
        self.target_prediction_network.eval()
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.representation_network.parameters()) +
            list(self.dynamics_network.parameters()) +
            list(self.prediction_network.parameters()),
            lr=1e-3
        )
        
        self.cur_bellman_err = 0
        self.last_avg_bellman_loss = 0.0
        self.mcts_simulations = params.get('mcts_simulations', 20)
        self.planning_depth = params.get('planning_depth', 5)
        
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
        self.game_history = []
    
    def prepare_state_representation(self, state):
        """Prepare state representation"""
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
        """Select action using planning"""
        self._user_request_slots = [
            s for s in state['current_slots']['request_slots'].keys()
            if s != 'taskcomplete'
        ]
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        
        if isinstance(self.action, torch.Tensor):
            self.action = int(self.action.view(-1)[0].item())
        
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
    
    def run_policy(self, representation):
        """Run policy with optional planning"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        if self.warm_start == 1:
            if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                self.warm_start = 2
            return self.rule_policy()
        else:
            # Use MCTS for planning
            if self.predict_mode or random.random() < 0.5:  # Mix MCTS with network policy
                return self.mcts_planning(representation)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(representation)
                    latent_state = self.representation_network(state_tensor)
                    policy, _ = self.prediction_network(latent_state)
                    action = torch.argmax(policy, dim=1).item()
                return action
    
    def mcts_planning(self, observation):
        """Monte Carlo Tree Search for action selection"""
        with torch.no_grad():
            root_latent_state = self.representation_network(torch.FloatTensor(observation))
            root_policy, root_value = self.prediction_network(root_latent_state)
            
        root = MCTSNode(prior=1.0)
        root_policy_probs = torch.softmax(root_policy, dim=1)[0].cpu().numpy()
        
        for action in range(self.num_actions):
            root.children[action] = MCTSNode(prior=root_policy_probs[action])
        
        for _ in range(self.mcts_simulations):
            self._mcts_simulate(root, root_latent_state)
        
        # Choose action with highest visit count
        visit_counts = [root.children[a].visit_count for a in range(self.num_actions)]
        return np.argmax(visit_counts)
    
    def _mcts_simulate(self, node, latent_state):
        """One simulation of MCTS"""
        search_path = [node]
        path_rewards = []
        
        for depth in range(self.planning_depth):
            if not node.children:
                break
            
            # Select action with highest UCB score
            actions = list(node.children.keys())
            ucb_scores = [node.children[a].ucb_score(node.visit_count) for a in actions]
            best_action = actions[np.argmax(ucb_scores)]
            
            # Get next state and reward
            action_one_hot = torch.zeros(1, self.num_actions).to(DEVICE)
            action_one_hot[0, best_action] = 1.0
            
            with torch.no_grad():
                latent_state, reward = self.dynamics_network(latent_state, action_one_hot)
                _, value = self.prediction_network(latent_state)
            
            path_rewards.append(reward.item())
            node = node.children[best_action]
            search_path.append(node)
        
        # Backup value through path
        bootstrap_value = 0
        with torch.no_grad():
            _, bootstrap_value = self.target_prediction_network(latent_state)
        
        cumulative_value = bootstrap_value.item()
        for t in range(len(path_rewards) - 1, -1, -1):
            cumulative_value = path_rewards[t] + self.gamma * cumulative_value
            search_path[t + 1].value_sum += cumulative_value
            search_path[t + 1].visit_count += 1
    
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
        
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        
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
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor(np.array([t[3].squeeze() for t in batch]))
        terminals = torch.FloatTensor([t[4] for t in batch])
        
        return states, actions, rewards, next_states, terminals
    
    def train(self, batch_size, num_epochs):
        """Train MuZero networks"""
        if not self.running_experience_pool or len(self.running_experience_pool) < batch_size:
            return
        
        for epoch in range(num_epochs):
            states, actions, rewards, next_states, terminals = self.sample_from_buffer(batch_size)
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            terminals = terminals.to(DEVICE)
            
            # Encode states
            latent_states = self.representation_network(states)
            next_latent_states = self.representation_network(next_states)
            
            # Predict policy and value
            policy, values = self.prediction_network(latent_states)
            
            # Dynamics prediction
            action_one_hot = torch.zeros(batch_size, self.num_actions).to(DEVICE)
            action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
            predicted_next_states, predicted_rewards = self.dynamics_network(latent_states, action_one_hot)
            
            # Target values
            with torch.no_grad():
                _, next_values = self.target_prediction_network(next_latent_states)
                target_values = rewards.unsqueeze(1) + (1 - terminals.unsqueeze(1)) * self.gamma * next_values
            
            # Value loss
            value_loss = torch.nn.functional.smooth_l1_loss(values, target_values)
            
            # Policy loss (maximize log probability of taken action)
            action_one_hot = torch.zeros(batch_size, self.num_actions).to(DEVICE)
            action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
            policy_loss = -torch.sum(action_one_hot * torch.log_softmax(policy, dim=1), dim=1).mean()
            
            # Reward prediction loss
            reward_loss = torch.nn.functional.smooth_l1_loss(predicted_rewards, rewards.unsqueeze(1))
            
            # State consistency loss
            state_consistency_loss = torch.nn.functional.mse_loss(
                predicted_next_states,
                next_latent_states.detach()
            )
            
            total_loss = value_loss + policy_loss + 0.25 * reward_loss + 0.25 * state_consistency_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            self.last_avg_bellman_loss = total_loss.item()
    
    def reset_dqn_target(self):
        """Update target networks"""
        self.target_prediction_network.load_state_dict(self.prediction_network.state_dict())
    
    def set_user_planning(self, user_planning):
        """Set world model for planning"""
        self.world_model = user_planning
    
    def save(self, path):
        """Save model"""
        torch.save({
            'representation_network': self.representation_network.state_dict(),
            'dynamics_network': self.dynamics_network.state_dict(),
            'prediction_network': self.prediction_network.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.representation_network.load_state_dict(checkpoint['representation_network'])
        self.dynamics_network.load_state_dict(checkpoint['dynamics_network'])
        self.prediction_network.load_state_dict(checkpoint['prediction_network'])
