from .usersim import UserSimulator
import argparse, json, random, copy, sys
import numpy as np
from .user_model import SimulatorModel
from collections import namedtuple, deque
from deep_dialog import dialog_config

import torch
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'agent_action', 'next_state', 'reward', 'term', 'user_action'))


class ModelBasedSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.feasible_actions_users = dialog_config.feasible_actions_users
        self.num_actions = len(self.feasible_actions)
        self.num_actions_user = len(self.feasible_actions_users)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 9 * self.slot_cardinality + 3 + self.max_turn

        # Precomputed offsets for prepare_state_representation (avoids 11 small array allocs per call)
        A, S, T = self.act_cardinality, self.slot_cardinality, self.max_turn
        self._sr_off = {
            'user_act':      0,
            'user_inform':   A,
            'user_request':  A + S,
            'agent_act':     A + 2*S,
            'agent_inform':  2*A + 2*S,
            'agent_request': 2*A + 3*S,
            'cur_slots':     2*A + 4*S,
            'turn':          2*A + 5*S,
            'turn_onehot':   2*A + 5*S + 1,
            'kb_binary':     2*A + 5*S + 1 + T,
            'kb_count':      2*A + 6*S + 2 + T,
        }
        # state_dimension excludes the 2*S goal appended by register_experience_replay_tuple

        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)

        self.learning_phase = params['learning_phase']
        self.hidden_size = params['hidden_size']

        self.training_examples = deque(maxlen=self.experience_replay_pool_size)

        self.predict_model = True

        self.model = SimulatorModel(self.num_actions, self.hidden_size, self.state_dimension, self.num_actions_user, 1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.goal = self._sample_goal(self.start_set)
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(list(dialog_config.start_dia_acts.keys()))

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot:
                    continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
            self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        return self.sample_goal

    def prepare_user_goal_representation(self, user_goal):
        S = self.slot_cardinality
        rep = np.zeros(2 * S, dtype=np.float32)
        for s in user_goal['request_slots']:
            rep[self.slot_set[s.strip()]] = 1.0
        for s in user_goal['inform_slots']:
            rep[S + self.slot_set[s.strip()]] = 1.0
        self.user_goal_representation = rep.reshape(1, -1)
        return self.user_goal_representation

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        batch = random.choices(self.training_examples, k=batch_size)
        np_batch = [np.vstack([b[x] for b in batch]) for x in range(len(Transition._fields))]
        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=1):
        """
        Train the world model with all the accumulated examples
        :param batch_size: self-explained
        :param num_batches: self-explained
        :return: None
        """
        self.total_loss = 0
        if len(self.training_examples) == 0:
            return
        for iter_batch in range(num_batches):
            for iter in range(len(self.training_examples) // batch_size):
                self.optimizer.zero_grad()

                batch = self.sample_from_buffer(batch_size)
                state = torch.FloatTensor(batch.state)
                action = torch.LongTensor(batch.agent_action)
                reward = torch.FloatTensor(batch.reward)
                term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
                user_action = torch.LongTensor(batch.user_action).squeeze(1)

                reward_, term_, user_action_ = self.model(state, action)

                loss = F.mse_loss(reward_, reward) + \
                       F.binary_cross_entropy_with_logits(term_, term) + \
                       F.nll_loss(user_action_, user_action)
                loss.backward()

                self.optimizer.step()
                self.total_loss += loss.item()

            denom = max(1.0, float(len(self.training_examples)) / float(batch_size))
            print("Total cost for user modeling: %.4f, training replay pool %s" % (
                float(self.total_loss) / denom, len(self.training_examples)))

    def train_by_iter(self, batch_size=1, num_batches=1):
        """
        Train the model with num_batches examples.
        :param batch_size:
        :param num_batches:
        :return: None
        """
        self.total_loss = 0
        for iter_batch in range(num_batches):
            self.optimizer.zero_grad()
            batch = self.sample_from_buffer(batch_size)
            state = torch.FloatTensor(batch.state)
            action = torch.LongTensor(batch.agent_action)
            reward = torch.FloatTensor(batch.reward)
            term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
            user_action = torch.LongTensor(batch.user_action).squeeze(1)

            reward_, term_, user_action_ = self.model(state, action)

            loss = F.mse_loss(reward_, reward) + \
                   F.binary_cross_entropy_with_logits(term_, term) + \
                   F.nll_loss(user_action_, user_action)
            loss.backward()

            self.optimizer.step()
            self.total_loss = loss.item()

            print ("Total cost for user modeling: %.4f, training replay pool %s" % (
                float(self.total_loss), len(self.training_examples)))

    def next(self, s, a):
        """
        Provide
        :param s: state representation from tracker
        :param a: last action from agent
        :return: next user action, termination and reward predicted by world model
        """

        self.state['turn'] += 2
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn):
            reward = - self.max_turn
            term = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            self.add_nl_to_action(response_action)
            return response_action, term, reward

        s = self.prepare_state_representation(s)
        g = self.prepare_user_goal_representation(self.sample_goal)
        s = np.hstack([s, g])
        a_arr = np.asarray(a).reshape(-1, 1)
        reward, term, action = self.predict(torch.FloatTensor(s), torch.LongTensor(a_arr))
        action = action.item()
        reward = reward.item()
        term = term.item()
        action = copy.deepcopy(self.feasible_actions_users[action])

        if action['diaact'] == 'inform':
            if len(action['inform_slots'].keys()) > 0:
                slots = list(action['inform_slots'].keys())[0]
                if slots in self.sample_goal['inform_slots'].keys():
                    action['inform_slots'][slots] = self.sample_goal['inform_slots'][slots]
                else:
                    action['inform_slots'][slots] = dialog_config.I_DO_NOT_CARE

        response_action = action
        response_action['turn'] = self.state['turn']
        self.add_nl_to_action(response_action)

        term = term > 0.5

        if reward > 1:
            reward = 2 * self.max_turn
        elif reward < -1:
            reward = -self.max_turn
        else:
            reward = -1

        return response_action, term, reward

    def predict(self, s, a):
        return self.model.predict(s, a)

    def register_user_goal(self, goal):
        self.user_goal = goal

    def action_index(self, act_slot_response):
        """ Return the index of action """
        del act_slot_response['turn']
        del act_slot_response['nl']

        for i in act_slot_response['inform_slots'].keys():
            act_slot_response['inform_slots'][i] = 'PLACEHOLDER'

        # rule
        if act_slot_response['diaact'] == 'request': act_slot_response['inform_slots'] = {}
        if act_slot_response['diaact'] in ['thanks', 'deny', 'closing']: 
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {}
            
        for (i, action) in enumerate(self.feasible_actions_users):
            if act_slot_response == action:
                return i
        print (act_slot_response)
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, agent_a_t, s_tplus1, reward, term, user_a_t):
        """ Register feedback from the environment, to be stored as future training data for world model"""

        state_t_rep = self.prepare_state_representation(s_t)
        goal_rep = self.prepare_user_goal_representation(self.sample_goal)
        state_t_rep = np.hstack([state_t_rep, goal_rep])
        agent_action_t = agent_a_t
        user_action_t = user_a_t
        action_idx = self.action_index(copy.deepcopy(user_a_t))
        reward_t = reward
        term_t = term

        if reward_t > 1:
            reward_t = 1
        elif reward_t < -1:
            reward_t = -1
        elif reward_t == -1:
            reward_t = -0.1

        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example_for_user = (state_t_rep, agent_action_t, state_tplus1_rep, reward_t, term, action_idx)

        if self.predict_model:
            self.training_examples.append(training_example_for_user)


    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        S = self.slot_cardinality
        T = self.max_turn
        dim = 2 * self.act_cardinality + 7 * S + 3 + T
        rep = np.zeros(dim, dtype=np.float32)
        o = self._sr_off

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        rep[o['user_act'] + self.act_set[user_action['diaact']]] = 1.0
        for slot in user_action['inform_slots']:
            rep[o['user_inform'] + self.slot_set[slot]] = 1.0
        for slot in user_action['request_slots']:
            rep[o['user_request'] + self.slot_set[slot]] = 1.0

        if agent_last:
            rep[o['agent_act'] + self.act_set[agent_last['diaact']]] = 1.0
            for slot in agent_last['inform_slots']:
                rep[o['agent_inform'] + self.slot_set[slot]] = 1.0
            for slot in agent_last['request_slots']:
                rep[o['agent_request'] + self.slot_set[slot]] = 1.0

        for slot in current_slots['inform_slots']:
            rep[o['cur_slots'] + self.slot_set[slot]] = 1.0

        turn = state['turn']
        rep[o['turn']] = float(turn) / float(T)
        if turn < T:
            rep[o['turn_onehot'] + turn] = 1.0

        if isinstance(kb_results_dict, dict):
            kb_b, kb_c = o['kb_binary'], o['kb_count']
            if kb_results_dict.get('matching_all_constraints', 0) > 0:
                rep[kb_b + S] = 1.0
            if 'matching_all_constraints' in kb_results_dict:
                rep[kb_c + S] = float(kb_results_dict['matching_all_constraints']) / 100.0
            for slot, count in kb_results_dict.items():
                if slot in self.slot_set:
                    idx = self.slot_set[slot]
                    if count > 0:
                        rep[kb_b + idx] = 1.0
                    rep[kb_c + idx] = float(count) / 100.0

        self.final_representation = rep.reshape(1, -1)
        return self.final_representation
