"""
Build the restaurant-domain data files required by the Deep Dyna-Q dialog
system from the raw CamRest676 release (CamRest676.json, CamRestDB.json,
CamRestOTGY.json).

Outputs (written next to this script):
    camrest_kb.p                 Restaurant KB, dict[int -> {slot: value}]
    camrest_dict.p               Slot -> [possible values], used by error injection
    camrest_user_goals.p         List of user goals extracted from dialogues
    slot_set_camrest.txt         Slot vocabulary (one per line, blank line ends file)
    dia_acts_camrest.txt         Dialog-act vocabulary
    dia_act_nl_pairs_camrest.json  Template NLG pairs

Run from anywhere:
    python build_camrest_data.py
"""

import json
import os
import pickle
import re

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Restaurant ontology used by the rewritten dialog system.
# These lists are the source of truth -- dialog_config imports the same names.
# ---------------------------------------------------------------------------
INFORMABLE_SLOTS = ['area', 'food', 'pricerange']
REQUESTABLE_SLOTS = ['name', 'phone', 'address', 'postcode']
# Slots that may show up either as user constraint OR user request.
ALL_DOMAIN_SLOTS = sorted(set(INFORMABLE_SLOTS + REQUESTABLE_SLOTS))
# Non-domain control slots used by the dialog system internals.
CONTROL_SLOTS = ['taskcomplete', 'closing']

# Dialog acts used by the corpus + the system internals.
DIA_ACTS = [
    'request',
    'inform',
    'confirm_question',
    'confirm_answer',
    'greeting',
    'closing',
    'multiple_choice',
    'thanks',
    'welcome',
    'deny',
    'not_sure',
]


def _strip_header(path):
    """The CamRest JSON files start with a banner of '#' comment lines that
    is not legal JSON. Strip everything before the first '[' or '{'."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    m = re.search(r'[\[\{]', raw)
    if m is None:
        raise ValueError(f"No JSON content in {path}")
    return json.loads(raw[m.start():])


def load_raw():
    dialogs = _strip_header(os.path.join(HERE, 'CamRest676.json'))
    db = _strip_header(os.path.join(HERE, 'CamRestDB.json'))
    ontology = _strip_header(os.path.join(HERE, 'CamRestOTGY.json'))
    return dialogs, db, ontology


# ---------------------------------------------------------------------------
# KB and slot-value dictionary
# ---------------------------------------------------------------------------
def build_kb(db):
    """Materialize the restaurant KB as {int_id: {slot: value, ...}}."""
    kb = {}
    for i, entry in enumerate(db):
        rec = {}
        for slot in ALL_DOMAIN_SLOTS:
            if slot in entry and entry[slot] is not None:
                rec[slot] = str(entry[slot]).lower()
        kb[i] = rec
    return kb


def build_slot_dict(kb, ontology):
    """Slot -> list of possible values. Used by the user simulator for
    random slot corruption (slot_err_mode)."""
    d = {slot: set() for slot in ALL_DOMAIN_SLOTS}
    # Pull informable values from the ontology
    for slot, vals in ontology.get('informable', {}).items():
        if slot in d:
            for v in vals:
                d[slot].add(str(v).lower())
    # Add any extra values observed in the KB (covers requestable slots too)
    for rec in kb.values():
        for slot, val in rec.items():
            d[slot].add(str(val).lower())
    return {k: sorted(v) for k, v in d.items() if v}


# ---------------------------------------------------------------------------
# User goals
# ---------------------------------------------------------------------------
def _normalize_value(v):
    v = str(v).strip().lower()
    # CamRest676 uses 'dontcare' for indifference. The dialog system uses the
    # string in dialog_config.I_DO_NOT_CARE ("I do not care") for this.
    if v in ('dontcare', "don't care", 'do not care'):
        return 'I do not care'
    return v


def build_user_goals(dialogs):
    """Extract one user goal per dialogue.

    A user goal has the shape expected by usersims:
        {'diaact': 'request',
         'inform_slots': {slot: value, ...},   # user's constraints
         'request_slots': {slot: 'UNK', ...}}  # info the user is after
    """
    goals = []
    for dlg in dialogs:
        inform = {}
        request = {}
        for turn in dlg.get('dial', []):
            usr = turn.get('usr', {})
            for act in usr.get('slu', []):
                a = act.get('act')
                if a == 'inform':
                    for slot, val in act.get('slots', []):
                        if slot in INFORMABLE_SLOTS:
                            inform[slot] = _normalize_value(val)
                elif a == 'request':
                    # In CamRest676, request entries look like
                    # {"act": "request", "slots": [["slot", "address"]]}
                    # i.e. the first element is the literal token "slot" and
                    # the second is the requested slot name.
                    for key, val in act.get('slots', []):
                        if key == 'slot' and val in REQUESTABLE_SLOTS:
                            request[val] = 'UNK'
                        elif key in REQUESTABLE_SLOTS:
                            request[key] = 'UNK'
        # A goal is only useful if it has at least one constraint AND one
        # piece of info to ask for. Otherwise the dialogue is degenerate.
        if inform and request:
            goals.append({
                'diaact': 'request',
                'inform_slots': inform,
                'request_slots': request,
            })
    return goals


# ---------------------------------------------------------------------------
# NLG templates
# ---------------------------------------------------------------------------
def build_nlg_templates():
    """Build a comprehensive dia_act_nl_pairs.json for the restaurant domain.

    The NLG engine first looks up a template that matches (diaact, set of
    inform slots, set of request slots) exactly. We try to cover the
    combinations the rule-based agent and simulator will actually produce.
    A generic fallback in nlg.py handles anything not listed here.
    """
    # Per-slot phrases for inform_slots when a single slot is informed.
    single_inform = {
        'area':       ("It is in the $area$ area.",          "I want a restaurant in the $area$ area."),
        'food':       ("It serves $food$ food.",             "I'd like $food$ food."),
        'pricerange': ("It is $pricerange$.",                "I want a $pricerange$ restaurant."),
        'name':       ("$name$ is a great choice.",          "I am looking for $name$."),
        'phone':      ("The phone number is $phone$.",       "I want to know the phone number."),
        'address':    ("The address is $address$.",          "I want to know the address."),
        'postcode':   ("The postcode is $postcode$.",        "I want the postcode."),
    }
    single_request = {
        'area':       ("Which area would you prefer?",                  "What area is it in?"),
        'food':       ("What kind of food would you like?",             "What type of food does it serve?"),
        'pricerange': ("What price range are you looking for?",         "What is the price range?"),
        'name':       ("Is there a specific restaurant you have in mind?", "What is the name of the restaurant?"),
        'phone':      ("Do you need the phone number?",                 "Can I have the phone number?"),
        'address':    ("Do you need the address?",                      "Can I have the address?"),
        'postcode':   ("Do you need the postcode?",                     "Can I have the postcode?"),
    }

    pairs = {'dia_acts': {'inform': [], 'request': [], 'confirm_question': [],
                           'confirm_answer': [], 'thanks': [], 'closing': [],
                           'deny': [], 'welcome': [], 'multiple_choice': [],
                           'greeting': [], 'not_sure': []}}

    # ---- inform: single slot ----
    for slot, (agt, usr) in single_inform.items():
        pairs['dia_acts']['inform'].append({
            'request_slots': [],
            'inform_slots': [slot],
            'nl': {'agt': agt, 'usr': usr},
        })
    # ---- inform: taskcomplete (with any subset of informable slots) ----
    pairs['dia_acts']['inform'].append({
        'request_slots': [],
        'inform_slots': ['taskcomplete'],
        'nl': {'agt': "I have found a matching restaurant for you.",
               'usr': "Thanks, that works for me."},
    })
    # ---- inform: combinations seen in practice (agent informing several slots
    # together at task completion, e.g. name+phone+address) ----
    # Cover all subsets of requestable slots up to size 4 with the name slot
    # always present, since that's the typical "here is your restaurant" turn.
    from itertools import combinations
    other_req = [s for s in REQUESTABLE_SLOTS if s != 'name']
    for r in range(0, len(other_req) + 1):
        for combo in combinations(other_req, r):
            slots = ['name'] + list(combo)
            tmpl_agt = "I recommend $name$" + (
                " (" + ", ".join(f"{s}: ${s}$" for s in combo) + ")" if combo else "") + "."
            tmpl_usr = "I am interested in $name$."
            pairs['dia_acts']['inform'].append({
                'request_slots': [],
                'inform_slots': slots,
                'nl': {'agt': tmpl_agt, 'usr': tmpl_usr},
            })
    # Also include 2-combinations of arbitrary informable+requestable slots
    # the user simulator may produce in a single turn.
    for a in INFORMABLE_SLOTS:
        for b in INFORMABLE_SLOTS:
            if a >= b:
                continue
            pairs['dia_acts']['inform'].append({
                'request_slots': [],
                'inform_slots': [a, b],
                'nl': {'agt': f"It is ${a}$ and ${b}$.",
                       'usr': f"I want a ${a}$ ${b}$ restaurant."},
            })

    # ---- request: single slot, no inform slots ----
    for slot, (agt, usr) in single_request.items():
        pairs['dia_acts']['request'].append({
            'request_slots': [slot],
            'inform_slots': [],
            'nl': {'agt': agt, 'usr': usr},
        })

    # ---- request with one inform slot (user asks for X while telling Y) ----
    for r_slot in REQUESTABLE_SLOTS:
        for i_slot in INFORMABLE_SLOTS:
            pairs['dia_acts']['request'].append({
                'request_slots': [r_slot],
                'inform_slots': [i_slot],
                'nl': {
                    'agt': f"What ${r_slot}$ for the ${i_slot}$ restaurant?",
                    'usr': f"Can I get the ${r_slot}$ for the ${i_slot}$ restaurant?",
                },
            })

    # ---- thanks / closing / etc ----
    pairs['dia_acts']['thanks'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "You're welcome!", 'usr': "Thank you, goodbye."},
    })
    pairs['dia_acts']['closing'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Goodbye.", 'usr': "Goodbye."},
    })
    pairs['dia_acts']['deny'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Sorry, that is not correct.", 'usr': "No, that is not what I want."},
    })
    pairs['dia_acts']['confirm_question'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Could you confirm your request?", 'usr': "Are you sure?"},
    })
    pairs['dia_acts']['confirm_answer'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Yes, confirmed.", 'usr': "Yes, that's right."},
    })
    pairs['dia_acts']['welcome'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Welcome!", 'usr': "Hi."},
    })
    pairs['dia_acts']['greeting'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Hello, how can I help you?", 'usr': "Hello."},
    })
    pairs['dia_acts']['multiple_choice'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "Which one would you like?", 'usr': "Either is fine."},
    })
    pairs['dia_acts']['not_sure'].append({
        'request_slots': [], 'inform_slots': [],
        'nl': {'agt': "I'm not sure.", 'usr': "I'm not sure."},
    })

    return pairs


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------
def write_lines(path, items):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for item in items:
            f.write(item + '\n')


def write_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def main():
    dialogs, db, ontology = load_raw()
    print(f"Loaded: {len(dialogs)} dialogues, {len(db)} restaurants")

    kb = build_kb(db)
    slot_dict = build_slot_dict(kb, ontology)
    goals = build_user_goals(dialogs)
    print(f"Built KB with {len(kb)} entries, dictionary covers {len(slot_dict)} slots, "
          f"{len(goals)} usable user goals")

    write_pickle(os.path.join(HERE, 'camrest_kb.p'), kb)
    write_pickle(os.path.join(HERE, 'camrest_dict.p'), slot_dict)
    write_pickle(os.path.join(HERE, 'camrest_user_goals.p'), goals)

    # The dialog system reads slot/act sets via text_to_dict in
    # deep_dialog/dialog_system/dict_reader.py, which expects one token per
    # line and tolerates a trailing blank line.
    all_slots = ALL_DOMAIN_SLOTS + CONTROL_SLOTS
    write_lines(os.path.join(HERE, 'slot_set_camrest.txt'), all_slots)
    write_lines(os.path.join(HERE, 'dia_acts_camrest.txt'), DIA_ACTS)

    nlg = build_nlg_templates()
    with open(os.path.join(HERE, 'dia_act_nl_pairs_camrest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(nlg, f, indent=2)

    print("Done. Files written to", HERE)


if __name__ == '__main__':
    main()
