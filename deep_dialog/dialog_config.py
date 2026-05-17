"""
Dialog configuration for the CamRest676 restaurant domain.

Informable slots (user constraints): area, food, pricerange
Requestable slots (info user can ask for): name, phone, address, postcode
Control slots used by the dialog system: taskcomplete, closing
"""

################################################################################
#  Domain slots
################################################################################
INFORMABLE_SLOTS = ['area', 'food', 'pricerange']
REQUESTABLE_SLOTS = ['name', 'phone', 'address', 'postcode']

# Slots the system can request from the user (i.e. user's constraints).
sys_request_slots = list(INFORMABLE_SLOTS)

# Slots the system can inform back to the user. Includes the "taskcomplete"
# control slot, whose inform value is used as the success/fail signal.
sys_inform_slots = list(REQUESTABLE_SLOTS) + list(INFORMABLE_SLOTS) + ['taskcomplete']

# Slots the user simulator may inform the agent of.
sys_inform_slots_for_user = list(INFORMABLE_SLOTS) + list(REQUESTABLE_SLOTS) + ['closing']

# Slots the user simulator may request from the agent.
sys_request_slots_for_user = list(REQUESTABLE_SLOTS)

# Dialog acts the user can open the conversation with.
start_dia_acts = {
    'request': list(REQUESTABLE_SLOTS),
}

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
# Sentinel returned by KBHelper when at least one DB row matches the user's
# constraints. Kept under the old name for compatibility with existing code.
TICKET_AVAILABLE = 'Match Available'

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for NL; 1 for dia-act; 2 for both
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   Slots the KB lookup must IGNORE when filtering rows (control slots).
################################################################################
NON_KB_SLOTS = {'taskcomplete', 'closing'}

################################################################################
#   Feasible actions for the RL agent.
################################################################################
feasible_actions = [
    {'diaact': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
    {'diaact': "confirm_answer",   'inform_slots': {}, 'request_slots': {}},
    {'diaact': "thanks",           'inform_slots': {}, 'request_slots': {}},
    {'diaact': "deny",             'inform_slots': {}, 'request_slots': {}},
]
for slot in sys_inform_slots:
    feasible_actions.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})
for slot in sys_request_slots:
    feasible_actions.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})

################################################################################
#   Feasible actions for the (model-based) user simulator.
################################################################################
feasible_actions_users = [
    {'diaact': "thanks",         'inform_slots': {}, 'request_slots': {}},
    {'diaact': "deny",           'inform_slots': {}, 'request_slots': {}},
    {'diaact': "closing",        'inform_slots': {}, 'request_slots': {}},
    {'diaact': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
]
for slot in sys_inform_slots_for_user:
    feasible_actions_users.append({'diaact': 'inform', 'inform_slots': {slot: "PLACEHOLDER"}, 'request_slots': {}})
for slot in sys_request_slots_for_user:
    feasible_actions_users.append({'diaact': 'request', 'inform_slots': {}, 'request_slots': {slot: "UNK"}})
feasible_actions_users.append({'diaact': 'inform', 'inform_slots': {}, 'request_slots': {}})
