import argparse
import sys
from amrlib.models.parse_xfm.inference import Inference
from daide2eng.utils import gen_English, is_daide,create_daide_grammar
sys.path.insert(0, '../AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide_LLM import AMR
import regex
import json
import glob
import os
def main():
    AMRparser = argparse.ArgumentParser()
    AMRparser.add_argument('--document',type=str)

    args = AMRparser.parse_args()
    json_file: str = args.document
    with open(json_file, 'r') as file:
            data = json.load(file)
    New_dict = dict()
    for i in data:
        print(i)
        for s , j in data[i].items():
            if j:
                j = remove_duplicates_corrected(j)
            for k in range(len(j)):
                graph, daide_status,daide_s = ENG_AMR(j[k]['message'],j[k]['sender'].capitalize(),j[k]['recipient'].capitalize())
                j[k]['daide_message'] = daide_s
                j[k]['amr_message'] = graph
                j[k]['daide_status'] = daide_status
            New_dict[i] = {s:j}
    name, ext = os.path.splitext(json_file)
    new_file_name = f"{name}_updated{ext}"
    with open(new_file_name, 'w') as file:
        json.dump(New_dict, file, indent=4)
    print(f"Processed {json_file} into {new_file_name}")

# This function are used to remove duplicated messages except time sent difference.
def remove_duplicates_corrected(messages):
    seen = set()
    unique_messages = []

    for message in messages:
        # Creating an identifier by converting lists to tuples and ignoring 'time_sent'
        identifier = tuple(
            tuple(message[key]) if isinstance(message[key], list) else message[key]
            for key in message if key != 'time_sent'
        )

        if identifier not in seen:
            seen.add(identifier)
            unique_messages.append(message)

    return unique_messages

def ENG_AMR(english,sender,recipient):
    num_beams   = 4
    batch_size  = 16
    device = 'cpu'
    model_dir  = 'personal/SEN_REC_MODEL/'
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)
    try:
      graph, daide_status,daide_s = eng_to_daide(english,sender,recipient,inference)
    except:
      graph, daide_status,daide_s = 'no-amr','NO-DAIDE',''
    return graph, daide_status,daide_s


def eng_to_daide(english,sender,recipient,inference):
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
            return 'No-DAIDE',''
        if amr_s2 == '(a / amr-empty)':
            daide_s, warnings = '', []
        else:
            daide_s, warnings = amr.amr_to_daide()
        daide_status = check_valid(daide_s)
        return graph,daide_status,daide_s

def check_valid(daide_sentence):
    grammar = create_daide_grammar(level=130)
    try:
        parse_tree = grammar.parse(daide_sentence)
        Full = True
    except:
        Full = False
    if regex.search(r'[A-Z]{3}', daide_sentence):
        if regex.search(r'[a-z]', daide_sentence):
            daide_status = 'Partial-DAIDE'
        elif Full == False:
            daide_status = 'Para-DAIDE'
        else:
            daide_status = 'Full-DAIDE'
    else:
        daide_status = 'No-DAIDE'
    return daide_status


if __name__ == '__main__':
    main()