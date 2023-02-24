import json
import sys
sys.path.insert(0, '../../../amrlib')
#import amrlib
from   amrlib.models.parse_xfm.inference import Inference
from amrtodaide import AMR
import regex
def eng_to_daide():
    data = [json.loads(line) for line in open('RUSSIA_0.json', 'r')]
    test_data = ''
    count = 0
    num_beams   = 4
    batch_size  = 16
    device = 'cuda:0'
    model_dir  = '../../../amrlib/amrlib/data/model_parse_xfm/checkpoint-9920/'
    print('loading models')
    inference = amrlib.Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)
    print('generating')
    gen_graphs = inference.parse_sents(["Hey Austria! I know AT is one of the least successful alliances in the game, so I'm super down to work together here, especially if we can get Russia and Turkey fighting.","Hey Germany! Just reaching out to say I think our two nations tend to benefit a ton from working together, and I'm very down to work together if you'd be interested!"], disable_progress=False)
    for graph in gen_graphs:
        print(graph)
        amr = AMR()
        amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(graph)
        if amr_node:
            amr.root = amr_node
        amr_s2 = amr.amr_to_string()
        if amr_s2 == '(a / amr-empty)':
            daide_s, warnings = '', []
        else:
            daide_s, warnings = amr.amr_to_daide()
            print(daide_s)
        if regex.search(r'[A-Z]{3}', daide_s):
            if regex.search(r'[a-z]', daide_s):
                daide_status = 'Partial-DAIDE'
            elif warnings:
                daide_status = 'Para-DAIDE'
            else:
                daide_status = 'Full-DAIDE'
        else:
            daide_status = 'No-DAIDE'
        print(daide_status)


if __name__ == "__main__":
    eng_to_daide()