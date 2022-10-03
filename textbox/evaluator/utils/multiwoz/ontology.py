all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
db_domains = ['restaurant', 'hotel', 'attraction', 'train']

# original slot names in goals (including booking slots)
# requestable_slots_in_goals = {
#     "taxi": ["car type", "phone"],
#     "police": ["postcode", "address", "phone"],
#     "hospital": ["address", "phone", "postcode"],
#     "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
#     "attraction": ["entrance fee", "type", "address", "postcode", "phone", "area", "reference"],
#     "train": ["duration", "leaveat", "price", "arriveby", "id", "reference"],
#     "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
# }

# informable_slots_in_goals = {
#     "taxi": ["leaveat", "destination", "departure", "arriveby"],
#     "police": [],
#     "hospital": ["department"],
#     "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
#     "attraction": ["area", "type", "name"],
#     "train": ["destination", "day", "arriveby", "departure", "people", "leaveat"],
#     "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
# }

normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel":
    ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}
all_reqslot = [
    "car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food", "stars", "area",
    "reference", "time", "leave", "price", "arrive", "id"
]
# count: 17

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = [
    "type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name", "leave",
    "destination", "departure", "arrive", "department", "food", "time"
]
# count: 17

all_slots = all_reqslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
get_slot = {}
for s in all_slots:
    get_slot[s] = 1
# count: 24

# mapping slots in dialogue act to original goal slot names
da_abbr_to_slot_name = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

# slot merging: not used currently
# slot_name_to_value_token = {
#     'entrance fee': 'price',
#     'pricerange': 'price',
#     'arrive': 'time',
#     'leave': 'time',
#     'departure': 'name',
#     'destination': 'name',
#     'stay': 'count',
#     'people': 'count',
#     'stars': 'count',
# }
# dialog_act_dom = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital', 'general', 'booking']
dialog_acts = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    # 'booking': ['book', 'inform', 'nobook', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)
# print(all_acts)

dialog_act_params = {
    'inform': all_slots + ['choice', 'open'],
    'request': all_infslot + ['choice', 'price'],
    'nooffer': all_slots + ['choice'],
    'recommend': all_reqslot + ['choice', 'open'],
    'select': all_slots + ['choice'],
    # 'book': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'nobook': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'offerbook': all_slots + ['choice'],
    'offerbooked': all_slots + ['choice'],
    'reqmore': [],
    'welcome': [],
    'bye': [],
    'greet': [],
}

# dialog_acts = ['inform', 'request', 'nooffer', 'recommend', 'select', 'book', 'nobook', 'offerbook', 'offerbooked',
#                         'reqmore', 'welcome', 'bye', 'greet'] # thank
dialog_act_all_slots = all_slots + ['choice', 'open']
# act_span_vocab = ['['+i+']' for i in dialog_act_dom] + ['['+i+']' for i in dialog_acts] + all_slots

# value_token_in_resp = ['address', 'name', 'phone', 'postcode', 'area', 'food', 'pricerange', 'id',
#                                      'department', 'place', 'day', 'count', 'car']
# count: 12

# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}

# special slot tokens in responses
# not use at the momoent
slot_name_to_value_token = {
    # 'entrance fee': '[value_price]',
    # 'pricerange': '[value_price]',
    # 'arriveby': '[value_time]',
    # 'leaveat': '[value_time]',
    # 'departure': '[value_place]',
    # 'destination': '[value_place]',
    # 'stay': 'count',
    # 'people': 'count'
}

db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

special_tokens = [
    '<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>', '<eos_d>',
    '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>'
] + db_tokens

sos_eos_tokens = [
    '<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>', '<eos_d>', '<sos_u>',
    '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>'
]

eos_tokens = {
    'user': '<eos_u>',
    'user_delex': '<eos_u>',
    'resp': '<eos_r>',
    'resp_gen': '<eos_r>',
    'pv_resp': '<eos_r>',
    'bspn': '<eos_b>',
    'bspn_gen': '<eos_b>',
    'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>',
    'bsdx_gen': '<eos_b>',
    'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>',
    'aspn_gen': '<eos_a>',
    'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>',
    'dspn_gen': '<eos_d>',
    'pv_dspn': '<eos_d>'
}

sos_tokens = {
    'user': '<sos_u>',
    'user_delex': '<sos_u>',
    'resp': '<sos_r>',
    'resp_gen': '<sos_r>',
    'pv_resp': '<sos_r>',
    'bspn': '<sos_b>',
    'bspn_gen': '<sos_b>',
    'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>',
    'bsdx_gen': '<sos_b>',
    'pv_bsdx': '<sos_b>',
    'aspn': '<sos_a>',
    'aspn_gen': '<sos_a>',
    'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>',
    'dspn_gen': '<sos_d>',
    'pv_dspn': '<sos_d>'
}
