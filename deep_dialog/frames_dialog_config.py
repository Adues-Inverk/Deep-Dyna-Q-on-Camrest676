"""
Dialog configuration for the Frames travel / hotel-booking domain.

Informable slots (user constraints): dst_city, budget_range, n_adults, category
Requestable slots (info user can ask for): name, price, gst_rating, amenities
Control slots: taskcomplete, closing
"""

INFORMABLE_SLOTS = ['dst_city', 'budget_range', 'n_adults', 'category']
REQUESTABLE_SLOTS = ['name', 'price', 'gst_rating', 'amenities']

sys_request_slots         = list(INFORMABLE_SLOTS)
sys_inform_slots          = list(REQUESTABLE_SLOTS) + list(INFORMABLE_SLOTS) + ['taskcomplete']
sys_inform_slots_for_user = list(INFORMABLE_SLOTS) + list(REQUESTABLE_SLOTS) + ['closing']
sys_request_slots_for_user = list(REQUESTABLE_SLOTS)

start_dia_acts = {
    'request': list(REQUESTABLE_SLOTS),
}

FAILED_DIALOG    = -1
SUCCESS_DIALOG   =  1
NO_OUTCOME_YET   =  0

SUCCESS_REWARD   = 50
FAILURE_REWARD   =  0
PER_TURN_REWARD  =  0

I_DO_NOT_CARE    = "I do not care"
NO_VALUE_MATCH   = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Match Available'

CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

nlg_beam_size = 10
run_mode      = 0
auto_suggest  = 0

NON_KB_SLOTS = {'taskcomplete', 'closing'}

BUDGET_BINS = [("budget", 0, 1500), ("moderate", 1500, 2500), ("luxury", 2500, 1e9)]

feasible_actions = [
    {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
    {'diaact': "confirm_answer",   'inform_slots': {}, 'request_slots': {}},
    {'diaact': "thanks",           'inform_slots': {}, 'request_slots': {}},
    {'diaact': "deny",             'inform_slots': {}, 'request_slots': {}},
    {'diaact': "no_result",        'inform_slots': {}, 'request_slots': {}},
    {'diaact': "sorry",            'inform_slots': {}, 'request_slots': {}},
]
for slot in sys_inform_slots:
    feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})
for slot in sys_request_slots:
    feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})

feasible_actions_users = [
    {'diaact': "thanks",         'inform_slots': {}, 'request_slots': {}},
    {'diaact': "deny",           'inform_slots': {}, 'request_slots': {}},
    {'diaact': "closing",        'inform_slots': {}, 'request_slots': {}},
    {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
    {'diaact': "moreinfo",       'inform_slots': {}, 'request_slots': {}},
]
for slot in sys_inform_slots_for_user:
    feasible_actions_users.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})
for slot in sys_request_slots_for_user:
    feasible_actions_users.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})
feasible_actions_users.append({'diaact': 'inform', 'inform_slots': {}, 'request_slots': {}})
