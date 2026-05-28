import pickle, os
HERE = os.path.dirname(os.path.abspath(__file__))
kb    = pickle.load(open(os.path.join(HERE,'frames_kb.p'),         'rb'), encoding='latin1')
goals = pickle.load(open(os.path.join(HERE,'frames_user_goals.p'), 'rb'), encoding='latin1')

kb_cities   = set(h['dst_city'] for h in kb.values())
goal_cities = set(g['inform_slots'].get('dst_city','') for g in goals if 'dst_city' in g['inform_slots'])

overlap = kb_cities & goal_cities
print('KB cities:', len(kb_cities))
print('Goal cities:', len(goal_cities))
print('Overlap:', len(overlap))
print('Sample overlap:', sorted(overlap)[:10])
match   = sum(1 for g in goals if g['inform_slots'].get('dst_city','') in kb_cities)
no_match= sum(1 for g in goals if g['inform_slots'].get('dst_city','') not in kb_cities)
no_city = sum(1 for g in goals if 'dst_city' not in g['inform_slots'])
print('Goals with matching city:', match)
print('Goals with no matching city:', no_match)
print('Goals with no dst_city slot:', no_city)
