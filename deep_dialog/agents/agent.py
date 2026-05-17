from deep_dialog import dialog_config
class Agent:
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
    def initialize_episode(self):
        self.current_action = {}
        self.current_action["diaact"]= None
        self.current_action['inform_slots'] = {}
        self.current_action['request_slots'] = {}
        self.current_action['turn'] = 0

    def state_to_action(self, state, available_actions):
        act_slot_response = None
        act_slot_value_response = None
        return {"act_slot_response": act_slot_response, "act_slot_value_response": act_slot_value_response}
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, *args, **kwargs):
        pass
    def set_user_planning(self, user_planning):
        pass
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    def add_nl_to_action(self,agent_action,):
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'],
                                                                    'agt')  # self.nlg_model.translate_diaact(agent_action['act_slot_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'],
                                                                    'agt')  # self.nlg_model.translate_diaact(agent_action['act_slot_value_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence