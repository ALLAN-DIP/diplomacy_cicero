import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   amrlib.utils.logging import silence_penman, setup_logging, WARN, ERROR
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch,match_pair
from   amrlib.models.parse_xfm.inference import Inference
from   amrlib.models.parse_xfm.penman_serializer import load_and_serialize
import smatch

if __name__ == '__main__':

    model_dir  = '../personal/SEN_REC_MODEL'
    gold_fpath = os.path.join(model_dir, 'test-pred-gold.txt')
    pred_fpath = os.path.join(model_dir, 'test-pred.txt')

    gold_entries = get_entries(gold_fpath)
    test_entries = get_entries(pred_fpath)
    gold_entries2 = []
    test_entries2 = []
    index = []
    count = 0
    for i in range(len(gold_entries)):
        if gold_entries[i] == '(a / amr-empty)' or test_entries[i] == '(a / amr-empty)':
            count +=1
        else:
            index.append(i)
            gold_entries2.append(gold_entries[i])
            test_entries2.append(test_entries[i])
    
    store_path = os.path.join(model_dir, 'combine_smatch_amrlib.txt')
    combine_smatch = open(store_path, 'w')
    with open(gold_fpath) as f:
        data = f.read()
    with open(pred_fpath) as f:
        data2 = f.read()
    entries = []
    m = data.split('\n\n')
    n = data2.split('\n\n')
    for i in range(len(m)):
        print(i)
        precision, recall, f_score = compute_smatch(test_entries[i:i+1], gold_entries[i:i+1])
        combine_smatch.write('gold:\n'+m[i] + '\n')
        combine_smatch.write('pred:\n'+n[i] + '\n')
        combine_smatch.write('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score)+'\n\n')
        print(f_score)
    combine_smatch.close()
    precision1, recall1, f_score1 = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision1, recall1, f_score1))
    precision2, recall2, f_score2 = compute_smatch(test_entries2, gold_entries2)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision2, recall2, f_score2))


