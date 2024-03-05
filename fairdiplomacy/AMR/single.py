import argparse
import sys
from amrlib.models.parse_xfm.inference import Inference
from daide2eng.utils import gen_English, is_daide,create_daide_grammar
sys.path.insert(0, '../AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide_LLM import AMR
import regex
def main():
    AMRparser = argparse.ArgumentParser()
    AMRparser.add_argument('--english')
    AMRparser.add_argument('--version')
    AMRparser.add_argument('--sender')
    AMRparser.add_argument('--recipient')

    sentence = AMRparser.parse_args()
    if sentence.english:
      graph, daide_status,daide_s = ENG_AMR(sentence.english,sentence.sender,sentence.recipient)
    print(graph)
    print(daide_status)
    print(daide_s)

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