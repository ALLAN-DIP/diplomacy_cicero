#load AMR model
import sys
sys.path.insert(0, './fairdiplomacy_external/AMR/DAIDE/DiplomacyAMR/code')
sys.path.insert(0, './fairdiplomacy_external/AMR/penman')
sys.path.insert(0, './fairdiplomacy_external/AMR/')
from amrtodaide_LLM import AMR
import regex
from amrlib.amrlib.models.parse_xfm.inference import Inference

num_beams   = 4
batch_size  = 16

device = 'cuda:0'
# model_dir  = './drive/MyDrive/cicero_experiments/AMR/amrlib/amrlib/data/model_parse_xfm/checkpoint-9920/'
model_dir  = './fairdiplomacy_external/AMR/personal/SEN_REC_MODEL/'
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
        except RecursionError:
            return '(a / amr-empty)'
        return amr_s2

def parse_phase_messages_to_amr(phase_data):
    msgs = phase_data['messages'].copy()

    for msg in msgs:
        msg = parse_single_message_to_amr(msg)
        
    phase_data['messages'] = msgs
    return phase_data

def parse_single_message_to_amr(msg):
    try:
        amr_string = eng_to_amr(msg['message'],msg['sender'],msg['recipient'],inference)
    except:
        amr_string = '(a / amr-empty)'  
        
    msg['parsed-amr'] = amr_string
    return msg
      