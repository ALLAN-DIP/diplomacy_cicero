import json
import sys
sys.path.insert(0, '../../fairdiplomacy/AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide import AMR
# import penman
import regex
sys.path.insert(0, '../../fairdiplomacy/AMR/amrlib')
from amrlib.models.parse_xfm.inference import Inference
from checker1 import f_score
from utils import compute_accuracy
from daide2eng.utils import pre_process,is_daide,create_daide_grammar
def eng_to_daide(sentence,inference,sender,recipient):
    #data = [json.loads(line) for line in open('RUSSIA_0.json', 'r')]
    print('generating')
    grammar = create_daide_grammar(level=130)
    gen_graphs = inference.parse_sents(['SEN send to REC that '+sentence], disable_progress=False)
    for graph in gen_graphs:
        print(graph)
        graph = graph.replace('SEN',sender).replace('REC',recipient)
        print(graph)
        amr = AMR()
        amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(graph)
        if amr_node:
            amr.root = amr_node
        try:
            amr_s2 = amr.amr_to_string()
        except:
            return 'No-DAIDE',''
        if amr_s2 == '(a / amr-empty)':
            daide_s, warnings = '', []
        else:
            daide_s, warnings = amr.amr_to_daide()
        try:
            parse_tree = grammar.parse(daide_s)
            Full = True
        except:
            Full = False
        if regex.search(r'[A-Z]{3}', daide_s):
            if regex.search(r'[a-z]', daide_s):
                daide_status = 'Partial-DAIDE'
            elif Full == False:
                daide_status = 'Para-DAIDE'
            else:
                daide_status = 'Full-DAIDE'
        else:
            daide_status = 'No-DAIDE'


    return daide_s,daide_status

if __name__ == "__main__":
    test_data = ''
    count = 0
    num_beams   = 4
    batch_size  = 16
    device = 'cpu'

    model_dir  = '../../fairdiplomacy/AMR/amrlib/amrlib/data/model_parse_xfm/checkpoint-4144/'
    print('loading models')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

    with open('/work/09256/yanze/ls6/diplomacy_cicero/fairdiplomacy/AMR/DAIDE/DiplomacyAMR/annotations/dip_daide_full.json','r') as file:
        data = json.load(file)

    with open("best_model4.txt", "w") as file:
        # Redirect stdout to the file
        sys.stdout = file

        # Your code here
        print("This will be written to output.txt")
        count = 0
        count2 = 0 
        overall_F = 0
        overall_F2 = 0
        overall_F_noremove = 0
        overall_F2_noremove = 0
        for message in data:
            print('=' * 50)
            msg = message["msg"]
            daide = message["daide"]
            if "sender" in message.keys():
                count += 1
                print(count)
                sender = message["sender"]
                recipient = message["recipient"]
                sentence = msg
                print(sentence)
                daide_s, daide_status = eng_to_daide(sentence,inference,sender.capitalize(),recipient.capitalize())
                print(f'gold daide is {daide}')
                print(f'translated daide is {daide_s}')
                print(daide_status)
                if daide_status == 'No-DAIDE':
                    count2 += 1
                else:
                    F = f_score(daide_s,daide,False)
                    overall_F = overall_F + F
                    print(f'better f_score is {F}')
                    F2 = compute_accuracy(daide,daide_s)
                    overall_F2 = overall_F2 +F2
                    print(f'original f_score is {F2}')
                F_noremove = f_score(daide_s,daide,False)
                overall_F_noremove = overall_F_noremove + F_noremove
                print(f'better f_score_noremove is {F_noremove}')
                F2_noremove = compute_accuracy(daide,daide_s)
                overall_F2_noremove = overall_F2_noremove +F2_noremove
                print(f'original f_score_noremove is {F2_noremove}')
        normal_count = count-count2
        average_F = overall_F/normal_count
        average_F2 = overall_F2/normal_count
        average_F_noremove = overall_F_noremove/count
        average_F2_noremove = overall_F2_noremove/count
        print(f'all messages is {count}')
        print(f'no daide messages is {count2}')
        print(f'daide messages is {normal_count}')
        print(f'average_F is {average_F}')
        print(f'average_F2 is {average_F2}')
        print(f'average_F_noremove is {average_F_noremove}')
        print(f'average_F2_noremove is {average_F2_noremove}')
    sys.stdout = sys.__stdout__
    #eng_to_daide("hhh")