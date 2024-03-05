### your AMR code
import sys
sys.path.insert(0, '../AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide_LLM import AMR
import regex
from amrlib.models.parse_xfm.inference import Inference
#from daide2eng.utils import gen_English, is_daide,create_daide_grammar
import os, json
POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

num_beams   = 4
batch_size  = 16

device = 'cuda:0'
model_dir  = 'personal/SEN_REC_MODEL/'
inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

def eng_to_amr(english,sender,recipient,inference):
    print('---------------------------')
    gen_graphs = inference.parse_sents(['SEN'+' send to '+'REC'+' that '+english.replace(sender,'SEN').replace(recipient,'REC')], disable_progress=False)
    for graph in gen_graphs:
        graph = graph.replace('SEN',sender).replace('REC',recipient)
        amr = AMR()
        amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(graph)
        if amr_node:
            amr.root = amr_node
        try:
            amr_s2 = amr.amr_to_string()
            return amr_s2
        except RecursionError:
            return '(a / amr-empty)'
        

def all_msgs(data):
    """
    get all messages in the game

    :param data: dictionary of game data after json.load
    :return: a dictionary of messages, with phase as key and list of messages as value
    """

    return data['message_history']

def msgs_from_to_focused_power(data, focused_power):
    """
    get all messages sent by focused_power in the game

    :param data: dictionary of game data after json.load
    :return: a dictionary of messages, with phase as key and list of messages as value
    """
    msgs = {}
    msgs_by_phase = all_msgs(data)

    for phase in msgs_by_phase:
        curr_phase_msgs = []

        for msg in msgs_by_phase[phase]:
            if msg['sender'] in focused_power and msg['recipient'] in focused_power:
                curr_phase_msgs.append(msg)

        msgs[phase] = curr_phase_msgs

    return msgs

def save_messages_amr_to_json(data):
    msg_by_phase_and_pair = dict()
    state_history = data['state_history']

    for phase in state_history:
      msg_by_phase_and_pair[phase] = dict()
      # pair of powers
      for power1 in POWERS:
        for power2 in POWERS:
          if power1 == power2:
            continue

          pair = '-'.join(sorted([power1, power2]))
          if pair in msg_by_phase_and_pair[phase]:
            continue
          # print(power1, power2)
          pair_msgs = msgs_from_to_focused_power(data, [power1, power2])[phase]
          # print(pair_msgs)
          pair_msgs_list = []
          hist_msg = ''
          for msg in pair_msgs:
            daide_msg = dict()
            try:
              daide_string = eng_to_amr(msg['message'],msg['sender'].capitalize(),msg['recipient'].capitalize(),inference)
            except:
                daide_string = '(a / amr-empty)'
            # print(daide_string)
            # msg -> DAIDE msg with timesent, phase, sender, recipient, message, DAIDE message, sender's unit, recipient's unit
            for key in ['time_sent', 'sender', 'recipient', 'phase', 'message','truth']:
                daide_msg[key] = msg[key]

            daide_msg['parsed-amr'] = daide_string

            pair_msgs_list.append(daide_msg)
            # hist_msg = copy.deepcopy(msg['message']) + ' '
          msg_by_phase_and_pair[phase][pair] = pair_msgs_list.copy()

    # File path where you want to save the JSON file
    file_path = f'./results/msg_amr_state_{data["game_id"]}.json'

    # Writing to a JSON file
    with open(file_path, 'w') as json_file:
      json.dump(msg_by_phase_and_pair, json_file, indent=4)


def main():

    #path to human games folder
    game_dir = '/home1/yanzewan/test_AMR/AMR/human_games/'
    # list of games to extract
    games = list(map(lambda x: game_dir + x, filter(lambda x: x.endswith('.json'), os.listdir(game_dir))))
    games = sorted(games, key=lambda x: x)

    for game in games:
        if 'AIGame' not in game:
          continue
        with open(game, 'r') as f:
            data = json.load(f)

        print(f'game_id: {data["game_id"]}')

        save_messages_amr_to_json(data)
main()