#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from   amrlib.evaluate.smatch_enhanced import compute_scores

# Score "nowiki" version, meaning the generated file should not have the :wiki tags added
GOLD='amrlib/data/tdata_gsii/test.txt.features.nowiki'
PRED='amrlib/data/model_parse_gsii/nomral200.pt.test_generated'

#GOLD='amrlib/data/tdata_gsiinew/testsmall.txt'
#PRED='amrlib/data/model_parse_gsii/testsmall.txt'

# Score with the test files with :wiki tags
#GOLD='amrlib/data/tdata_gsii/test.txt.features'
#PRED='amrlib/data/model_parse_gsii/normal_dip_200.pt.test_generated.wiki'

compute_scores(PRED, GOLD)
