"""
Rule-based user simulator for the Frames hotel-booking domain.

Goals have the shape:
  {'diaact': 'request',
   'inform_slots': {dst_city/budget_range/n_adults/category: value, ...},
   'request_slots': {name/price/gst_rating/amenities: 'UNK', ...}}
"""

from .usersim import UserSimulator
import random, copy
import deep_dialog.frames_dialog_config as dc


class FramesRuleSimulator(UserSimulator):

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set    = act_set
        self.slot_set   = slot_set
        self.start_set  = start_set

        self.max_turn             = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode        = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        self.simulator_run_mode   = params['simulator_run_mode']
        self.simulator_act_level  = params['simulator_act_level']
        self.learning_phase       = params['learning_phase']
        self.goal                 = {}
        self.sample_goal          = {}

    # ------------------------------------------------------------------
    def initialize_episode(self):
        self.state = {
            'history_slots': {},
            'inform_slots':  {},
            'request_slots': {},
            'rest_slots':    [],
            'turn':          0,
        }
        self.episode_over  = False
        self.dialog_status = dc.NO_OUTCOME_YET
        self.goal          = self._sample_goal()
        self.constraint_check = dc.CONSTRAINT_CHECK_FAILURE
        return self._sample_action()

    def _sample_goal(self):
        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        return self.sample_goal

    def get_goal(self):
        return self.sample_goal

    def _sample_action(self):
        self.state['diaact'] = random.choice(list(dc.start_dia_acts.keys()))

        if self.goal['inform_slots']:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]
            for slot in self.goal['inform_slots']:
                if slot != known_slot:
                    self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        request_slot_set = list(self.goal['request_slots'].keys())
        if request_slot_set:
            self.state['request_slots'][random.choice(request_slot_set)] = 'UNK'

        if not self.state['request_slots']:
            self.state['diaact'] = 'inform'

        self.episode_over = self.state['diaact'] in ('thanks', 'closing')

        action = {
            'diaact':       self.state['diaact'],
            'inform_slots': self.state['inform_slots'],
            'request_slots': self.state['request_slots'],
            'turn':         self.state['turn'],
        }
        self.add_nl_to_action(action)
        return action

    # ------------------------------------------------------------------
    def next(self, system_action):
        self.state['turn'] += 2
        self.episode_over  = False
        self.dialog_status = dc.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        if self.max_turn > 0 and self.state['turn'] > self.max_turn:
            self.dialog_status = dc.FAILED_DIALOG
            self.episode_over  = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self._response_inform(system_action)
            elif sys_act == "request":
                self._response_request(system_action)
            elif sys_act == "thanks":
                self._response_thanks(system_action)
            elif sys_act == "confirm_answer":
                self._response_confirm_answer(system_action)
            elif sys_act in ("no_result", "sorry"):
                # Agent couldn't find a match — user relaxes a constraint
                self._response_no_result(system_action)
            elif sys_act in ("closing", "greeting"):
                if sys_act == "closing":
                    self.episode_over = True
                    self.state['diaact'] = "thanks"
                    self.state['request_slots'].clear()

        if self.state['diaact'] == "thanks":
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()

        # When informing, request_slots must be empty so action_index can match
        # the action against feasible_actions_users (which never mixes both).
        if self.state['diaact'] == "inform":
            self.state['request_slots'].clear()

        self._corrupt(self.state)

        response = {
            'diaact':       self.state['diaact'],
            'inform_slots': self.state['inform_slots'],
            'request_slots': self.state['request_slots'],
            'turn':         self.state['turn'],
            'nl':           "",
        }
        self.add_nl_to_action(response)
        return response, self.episode_over, self.dialog_status

    # ------------------------------------------------------------------
    def _response_inform(self, system_action):
        if 'taskcomplete' in system_action['inform_slots']:
            self.state['diaact'] = "thanks"
            self.constraint_check = dc.CONSTRAINT_CHECK_SUCCESS

            if system_action['inform_slots']['taskcomplete'] == dc.NO_VALUE_MATCH:
                self.state['history_slots']['taskcomplete'] = dc.NO_VALUE_MATCH
                self.state['request_slots'].clear()

            for slot in self.goal['inform_slots']:
                if (slot not in system_action['inform_slots'] or
                        self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
                    self.state['diaact'] = "deny"
                    self.state['request_slots'].clear()
                    self.state['inform_slots'].clear()
                    self.constraint_check = dc.CONSTRAINT_CHECK_FAILURE
                    break

            self.state['request_slots'].clear()
        else:
            for slot in system_action['inform_slots']:
                self.state['history_slots'][slot] = system_action['inform_slots'][slot]

                if slot in self.goal['inform_slots']:
                    if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
                        if slot in self.state['rest_slots']:
                            self.state['rest_slots'].remove(slot)
                        if self.state['request_slots']:
                            self.state['diaact'] = "request"
                        elif self.state['rest_slots']:
                            self._pick_from_rest()
                        else:
                            self.state['diaact'] = "thanks"
                            self.state['request_slots'].clear()
                    else:
                        self.state['diaact'] = "inform"
                        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                        if slot in self.state['rest_slots']:
                            self.state['rest_slots'].remove(slot)
                        self.state['request_slots'].clear()
                else:
                    if slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(slot)
                    if slot in self.state['request_slots']:
                        del self.state['request_slots'][slot]

                    if self.state['request_slots']:
                        self.state['diaact'] = "request"
                    elif self.state['rest_slots']:
                        self._pick_from_rest()
                    else:
                        self.state['diaact'] = "thanks"
                        self.state['request_slots'].clear()

    def _response_request(self, system_action):
        if not system_action['request_slots']:
            if self.state['rest_slots']:
                self._pick_from_rest()
            return

        slot = list(system_action['request_slots'].keys())[0]
        if slot in self.goal['inform_slots']:
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
            self.state['diaact'] = "inform"
            if slot in self.state['rest_slots']:
                self.state['rest_slots'].remove(slot)
            self.state['request_slots'].clear()
        elif slot in self.goal['request_slots'] and slot in self.state['history_slots']:
            self.state['inform_slots'][slot] = self.state['history_slots'][slot]
            self.state['request_slots'].clear()
            self.state['diaact'] = "inform"
        elif slot in self.goal['request_slots'] and slot in self.state['rest_slots']:
            self.state['request_slots'].clear()
            self.state['diaact'] = "request"
            self.state['request_slots'][slot] = "UNK"
        else:
            if not self.state['request_slots'] and not self.state['rest_slots']:
                self.state['diaact'] = "thanks"
            else:
                self.state['diaact'] = "inform"
            self.state['inform_slots'][slot] = dc.I_DO_NOT_CARE
            self.state['request_slots'].clear()

    def _response_thanks(self, system_action):
        self.episode_over  = True
        self.dialog_status = dc.SUCCESS_DIALOG

        if self.state['request_slots'] or self.state['rest_slots']:
            self.dialog_status = dc.FAILED_DIALOG

        for slot, val in self.state['history_slots'].items():
            if val == dc.NO_VALUE_MATCH:
                self.dialog_status = dc.FAILED_DIALOG
            if slot in self.goal['inform_slots']:
                if val != self.goal['inform_slots'][slot]:
                    self.dialog_status = dc.FAILED_DIALOG

        if 'taskcomplete' in system_action['inform_slots']:
            if system_action['inform_slots']['taskcomplete'] == dc.NO_VALUE_MATCH:
                self.dialog_status = dc.FAILED_DIALOG

        if self.constraint_check == dc.CONSTRAINT_CHECK_FAILURE:
            self.dialog_status = dc.FAILED_DIALOG

    def _response_confirm_answer(self, system_action):
        if self.state['rest_slots']:
            slot = random.choice(self.state['rest_slots'])
            if slot in self.goal['request_slots']:
                self.state['request_slots'].clear()
                self.state['diaact'] = "request"
                self.state['request_slots'][slot] = "UNK"
            elif slot in self.goal['inform_slots']:
                self.state['diaact'] = "inform"
                self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                self.state['request_slots'].clear()
                if slot in self.state['rest_slots']:
                    self.state['rest_slots'].remove(slot)
        else:
            self.state['diaact'] = "thanks"
            self.state['request_slots'].clear()

    def _response_no_result(self, _system_action):
        """Relax a random informable constraint, or close if no constraints left."""
        relaxable = [s for s in self.goal['inform_slots'] if s in self.state['history_slots']]
        if relaxable:
            slot = random.choice(relaxable)
            self.state['inform_slots'][slot] = dc.I_DO_NOT_CARE
            self.state['diaact'] = "inform"
        else:
            self.state['diaact'] = "thanks"
            self.state['request_slots'].clear()
            self.dialog_status = dc.FAILED_DIALOG
            self.episode_over  = True

    def _pick_from_rest(self):
        slot = random.choice(self.state['rest_slots'])
        if slot in self.goal['inform_slots']:
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
            self.state['diaact'] = "inform"
            self.state['rest_slots'].remove(slot)
        elif slot in self.goal['request_slots']:
            self.state['request_slots'][slot] = "UNK"
            self.state['diaact'] = "request"

    def _corrupt(self, state):
        for slot in list(state['inform_slots'].keys()):
            if random.random() < self.slot_err_probability:
                if self.slot_err_mode == 0:
                    if slot in self.movie_dict:
                        state['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                elif self.slot_err_mode == 3:
                    del state['inform_slots'][slot]

        if random.random() < self.intent_err_probability:
            state['diaact'] = random.choice(list(self.act_set.keys()))
