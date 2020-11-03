import sys, os, re, json
import itertools
import collections
import numpy as np
import pandas as pd
from heapq import nlargest
from jiant import *
from plots import *
from DataAll import *
from ResultAll import *
from utils import *

def main(args):
    dataAll = DataAll()
    resultAll = ResultAll(dataAll)
################################################# span1_span2_distance #################################################
    exp_layer_dict, first_neg_delta_dict, var_layer_dict, best_layer_dict, num_examples_dict = resultAll.get_all_thr_dict_values()
    plot_span_expected_layer(exp_layer_dict, "threshold", "expected layer", xlim=MAX_SPR_THRESHOLD_DISTANCE, barGraphToo=False)

    #getting the same value for the expected layer as in the article (3.62 and almost 4.29 and 2.713 for the NER task and 1.936 for nonterminals)
    exp_layer_all = resultAll.get_all_expected_layers()
################################################# EXP LAYER PER SPAN & SPAN DISTRIBUTION #############################################################
    span_exp_layer, span_prob = resultAll.get_all_span_exp_layer_prob()
    plot_span_expected_layer(span_exp_layer,"spans", "expected layer", xlim=3, barGraphToo=True)
    plot_span_expected_layer(span_prob, "spans", "span probability", xlim=3, barGraphToo=False)
################################################# imposing min_max Simpson Paradox #############################################################
    diffs_max_min = resultAll.impose_max_min()
    plot_diffs_max_min(diffs_max_min)
    plot_sympson_paradox(span_exp_layer, 'NER', 'co-reference')
############################################################# ALL COMBOS ##################################################################
    TCE_all, CDE_all, NDE_all, NIE_all = resultAll.get_all_TCE_CDE_NIE_NDE()
    exp_layer_diff_all = resultAll.get_all_expected_layer_diff()

    biggest_TCE_NDE_difference = resultAll.get_biggest_exp_layer_diff_NDE_diff(isTCE=True)
    biggest_exp_layer_NDE_difference = resultAll.get_biggest_exp_layer_diff_NDE_diff(isTCE=False)

    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_TCE_NDE_difference, noTCE=False, noNDE=False, noNIE=True, noExpLayerDiff=True)
    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_exp_layer_NDE_difference, noTCE=True, noNDE=False, noNIE=True, noExpLayerDiff=False)
    plot_all_CDE(CDE_all)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)