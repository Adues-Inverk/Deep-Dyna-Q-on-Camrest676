import copy,random
from deep_dialog import dialog_config
from .agent import Agent

class InformAgent(Agent):
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """

        self.state['turn'] += 2
        if self.current_slot_id < len(self.slot_set.keys()):
            slot = list(self.slot_set.keys())[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {},
                                 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
class RequestAllAgent(Agent):
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """

        self.state['turn'] += 2
        if self.current_slot_id < len(dialog_config.sys_request_slots):
            slot = dialog_config.sys_request_slots[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {},
                                 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RandomAgent(Agent):
    """ A simple agent to test the interface. This agent should choose actions randomly. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """

        self.state['turn'] += 2
        act_slot_response = copy.deepcopy(random.choice(dialog_config.feasible_actions))
        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class EchoAgent(Agent):
    """ A simple agent that informs all requested slots, then issues inform(taskcomplete) when the user stops making requests. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        user_action = state['user_action']

        self.state['turn'] += 2
        act_slot_response = {}
        act_slot_response['inform_slots'] = {}
        act_slot_response['request_slots'] = {}
        ########################################################################
        # find out if the user is requesting anything
        # if so, inform it
        ########################################################################
        if user_action['diaact'] == 'request':
            requested_slot = list(user_action['request_slots'].keys())[0]

            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'][requested_slot] = "PLACEHOLDER"
        else:
            act_slot_response['diaact'] = "thanks"

        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RequestBasicsAgent(Agent):
    """ Request the basic constraints, answer any user request_slots, then inform(taskcomplete) and thanks. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0
        # In the restaurant domain, the basic constraints to elicit are the
        # three informable slots (area / food / pricerange).
        self.request_set = list(dialog_config.sys_request_slots)
        self.phase = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """

        pending_user_requests = [
            s for s in state['current_slots']['request_slots'].keys()
            if s != 'taskcomplete'
        ]

        self.state['turn'] += 2
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1
            act_slot_response = {'diaact': "request", 'inform_slots': {},
                                 'request_slots': {slot: "UNK"},
                                 'turn': self.state['turn']}
        elif pending_user_requests:
            slot = pending_user_requests[0]
            act_slot_response = {'diaact': "inform", 'inform_slots': {slot: "PLACEHOLDER"},
                                 'request_slots': {}, 'turn': self.state['turn']}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}, 'turn': self.state['turn']}
            self.phase += 1
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {},
                                 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
