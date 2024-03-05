import sys
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch, compute_scores
from   amrlib.evaluate.smatch_enhanced import redirect_smatch_errors

GOLD='/work/09256/yanze/AMR_LLM/AMR/personal/improvement1/test-gold.txt'
PRED='/work/09256/yanze/AMR_LLM/AMR/personal/improvement1/test-pred.txt'

#redirect_smatch_errors('logs/score_smatch_errors.log')
# Run only the smatch score
if 0:
    gold_entries = get_entries(GOLD)
    test_entries = get_entries(PRED)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
# Compute enhanced scoring
else:
    compute_scores(GOLD, PRED)