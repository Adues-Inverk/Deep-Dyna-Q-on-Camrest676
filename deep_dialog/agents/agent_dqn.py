import random, copy, json
import pickle
import numpy as np
from collections import namedtuple, deque
from deep_dialog import dialog_config
from .agent import Agent
from deep_dialog.qlearning import DQN
import torch
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class PrioritizedReplayBuffer:
    """
    Circular buffer with proportional-priority sampling (PER, Schaul et al. 2016).

    New transitions always start with max_priority so they are guaranteed to be
    sampled at least once. After each training step the caller should call
    update_priorities() with the computed TD errors so future sampling is biased
    toward surprising / informative transitions.
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha          # priority exponent (0 = uniform, 1 = full PER)
        self._buf = [None] * capacity
        self._prios = np.ones(capacity, dtype=np.float64)
        self._pos = 0               # next write slot
        self._size = 0
        self._max_p = 1.0           # running max priority for new entries

    def append(self, transition):
        self._buf[self._pos] = transition
        self._prios[self._pos] = self._max_p
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, k, beta=0.4):
        """Return (samples, indices, IS-weights) for k transitions."""
        p = self._prios[:self._size] ** self.alpha
        probs = p / p.sum()
        indices = np.random.choice(self._size, size=k, p=probs, replace=True)
        # Importance-sampling weights: w_i = (N * P(i))^{-beta}, normalised
        w = (self._size * probs[indices]) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        return [self._buf[i] for i in indices], indices, w

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6
            self._prios[i] = p
            if p > self._max_p:
                self._max_p = p

    def clear(self):
        self._buf = [None] * self.capacity
        self._prios[:] = 1.0
        self._pos = 0
        self._size = 0
        self._max_p = 1.0

    def __len__(self):
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield self._buf[i]


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params.get('epsilon', 0.5)
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 10000)

        per_alpha = params.get('per_alpha', 0.6)
        self.experience_replay_pool = PrioritizedReplayBuffer(
            self.experience_replay_pool_size, alpha=per_alpha
        )
        self.experience_replay_pool_from_model = deque(maxlen=self.experience_replay_pool_size)

        self.hidden_size = params.get('dqn_hidden_size', 128)
        # γ=0.99 keeps the terminal reward signal visible across 15-20 dialog turns
        self.gamma = params.get('gamma', 0.99)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        self.kb_size = len(movie_dict) if movie_dict is not None else 1
        self.min_epsilon = params.get('min_epsilon', 0.05)
        self.epsilon_decay = params.get('epsilon_decay', 0.992)
        # Fraction of each training batch drawn from world-model experience
        self.world_model_weight = params.get('world_model_weight', 0.5)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 5 + self.max_turn

        # Precomputed offsets for prepare_state_representation
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
            'slot_count':    2*A + 5*S + 1,
            'turn_onehot':   2*A + 5*S + 3,
            'kb_binary':     2*A + 5*S + 3 + T,
            'kb_count':      2*A + 6*S + 4 + T,
        }

        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions, dropout_rate=0.25).to(DEVICE)
        self.target_dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions, dropout_rate=0.25).to(DEVICE)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        self.optimizer = optim.Adam(
            self.dqn.parameters(),
            lr=params.get('learning_rate', 1e-3),
            weight_decay=1e-4,
        )

        self.cur_bellman_err = 0
        self.last_avg_bellman_loss = 0.0
        self.update_counter = 0
        self.epoch_counter = 0

        # PER: beta anneals from initial toward 1.0 over training
        self._per_beta = params.get('per_beta', 0.4)
        # Soft target update (tau) is more stable than periodic hard copy
        self._target_tau = params.get('target_tau', 0.005)

        if params['trained_model_path'] is not None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = list(dialog_config.sys_request_slots)
        self._user_request_slots = []

    def state_to_action(self, state):
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

    def prepare_state_representation(self, state):
        rep = np.zeros(self.state_dimension, dtype=np.float32)
        o = self._sr_off
        S, T = self.slot_cardinality, self.max_turn

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
        rep[o['slot_count']]     = float(len(current_slots['inform_slots']))  / max(1.0, float(S))
        rep[o['slot_count'] + 1] = float(len(current_slots['request_slots'])) / max(1.0, float(S))
        if turn < T:
            rep[o['turn_onehot'] + turn] = 1.0

        if isinstance(kb_results_dict, dict):
            kb_b, kb_c = o['kb_binary'], o['kb_count']
            if kb_results_dict.get('matching_all_constraints', 0) > 0:
                rep[kb_b + S] = 1.0
            if 'matching_all_constraints' in kb_results_dict:
                rep[kb_c + S] = float(kb_results_dict['matching_all_constraints']) / float(self.kb_size)
            for slot, count in kb_results_dict.items():
                if slot in self.slot_set:
                    idx = self.slot_set[slot]
                    if count > 0:
                        rep[kb_b + idx] = 1.0
                    rep[kb_c + idx] = float(count) / float(self.kb_size)

        self.final_representation = rep.reshape(1, -1)
        return self.final_representation

    def run_policy(self, representation):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.DQN_policy(representation)

    def rule_policy(self):
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1
            act_slot_response = {'diaact': "request", 'inform_slots': {},
                                 'request_slots': {slot: "UNK"}}
        elif self._user_request_slots:
            slot = self._user_request_slots[0]
            act_slot_response = {'diaact': "inform", 'inform_slots': {slot: "PLACEHOLDER"},
                                 'request_slots': {}}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}
        return self.action_index(act_slot_response)

    def DQN_policy(self, state_representation):
        self.dqn.eval()
        with torch.no_grad():
            action = self.dqn.predict(torch.FloatTensor(state_representation).to(DEVICE))
        self.dqn.train()
        return action

    def action_index(self, act_slot_response):
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

    def train(self, batch_size=1, num_batches=1):
        """Train DQN with PER-sampled real experience + uniform world-model experience."""
        self.cur_bellman_err = 0.
        self.cur_bellman_err_planning = 0.

        n_real = len(self.experience_replay_pool)
        if n_real == 0:
            self.last_avg_bellman_loss = 0.0
            return

        n_model = len(self.experience_replay_pool_from_model)
        # Fraction of each batch from world model (capped by available samples)
        n_m = min(int(batch_size * self.world_model_weight), n_model)
        n_r = batch_size - n_m

        # Anneal IS-correction beta from initial value toward 1.0
        beta = min(1.0, self._per_beta + self.epoch_counter * 0.002)

        model_list = list(self.experience_replay_pool_from_model) if n_m > 0 else []
        n_iters = max(1, n_real // batch_size)

        self.dqn.train()
        for _ in range(num_batches):
            for __ in range(n_iters):
                self.optimizer.zero_grad()

                # --- Sample -------------------------------------------------------
                real_samples, real_idx, is_w = self.experience_replay_pool.sample(n_r, beta)

                if n_m > 0 and model_list:
                    model_samples = random.choices(model_list, k=n_m)
                    all_samples = real_samples + model_samples
                    all_w = np.concatenate([is_w, np.ones(n_m, dtype=np.float32)])
                else:
                    all_samples = real_samples
                    all_w = is_w

                # --- Build tensors ------------------------------------------------
                np_b = [np.vstack([s[x] for s in all_samples]) for x in range(len(Transition._fields))]
                b = Transition(*np_b)

                states      = torch.tensor(b.state,     dtype=torch.float32, device=DEVICE)
                actions     = torch.tensor(b.action,    dtype=torch.long,    device=DEVICE)
                rewards     = torch.tensor(b.reward,    dtype=torch.float32, device=DEVICE)
                next_states = torch.tensor(b.next_state, dtype=torch.float32, device=DEVICE)
                terms       = torch.tensor(np.asarray(b.term, dtype=np.float32), device=DEVICE)
                iw          = torch.tensor(all_w, dtype=torch.float32, device=DEVICE).unsqueeze(1)

                # --- Double-DQN forward ------------------------------------------
                state_value = self.dqn(states).gather(1, actions)

                with torch.no_grad():
                    self.dqn.eval()
                    next_acts = self.dqn(next_states).argmax(1).unsqueeze(1)
                    self.dqn.train()
                    next_val = self.target_dqn(next_states).gather(1, next_acts)

                target = rewards + self.gamma * next_val * (1 - terms)
                td_err = (state_value.detach() - target).abs()

                # IS-weighted Huber loss
                loss = (F.smooth_l1_loss(state_value, target, reduction='none') * iw).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update priorities for real-pool samples
                self.experience_replay_pool.update_priorities(
                    real_idx, td_err[:n_r].squeeze(1).cpu().numpy()
                )
                self.cur_bellman_err += loss.item()

                # Soft target-network update (τ step toward online weights)
                with torch.no_grad():
                    tau = self._target_tau
                    for tp, op in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                        tp.data.mul_(1.0 - tau).add_(tau * op.data)

        denom = max(1.0, n_real / float(batch_size))
        print("cur bellman err %.4f, experience replay pool %s, model replay pool %s" % (
            float(self.cur_bellman_err) / denom, n_real, n_m))

        self.last_avg_bellman_loss = self.cur_bellman_err / max(1, num_batches * n_iters)

        if not self.predict_mode and self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.epoch_counter += 1
        if self.epoch_counter > 100 and self.epoch_counter % 10 == 0:
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(5e-5, pg['lr'] * 0.98)

    ################################################################################
    #    Debug / persistence helpers
    ################################################################################
    def save_experience_replay_to_file(self, path):
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path,))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,)); print(e)

    def load_experience_replay_from_file(self, path):
        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print("Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    def save(self, filename):
        torch.save(self.dqn.state_dict(), filename)

    def load(self, filename):
        self.dqn.load_state_dict(torch.load(filename, map_location=DEVICE))

    def reset_dqn_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
