import sys, os, re, json
import itertools
import collections
import numpy as np
import pandas as pd
from heapq import nlargest
from jiant import *
from DataAll import *
from utils import *

class Result:
    def __init__(self, data_class, max_thr, at_most_least):
        # dict - for threshold, no dict - the general case - when looking at all instances
        self.data_class = data_class
        self.expected_layer, self.first_neg_delta, self.var_layer, self.best_num_layer = calc_expected_layer(data_class.data_df)
        self.expected_layer_dict, self.first_neg_delta_dict, self.var_layer_dict, self.best_layer_dict, self.num_examples_dict = self.get_exp_and_best_layer_dict(data_class.data_df, max_thr, at_most_least=at_most_least)

    def get_exp_and_best_layer_dict(self, df, max_threshold_distance, span=SPAN1_SPAN2_DIST, at_most_least=AT_MOST):
        # span = span type: span1 length (if only span1), distance between span1 and span 2 or the total length of span1
        #           to span2 (or vice versa). can be SPAN1_LEN, SPAN1_SPAN2_LEN or SPAN1_SPAN2_DIST
        # at_most_least = whether the close threshold (<=thr) or the far threshold (>=thr)
        exp_layer_dict, var_layer_dict, first_negative_delta_dict, best_layer_dict, num_examples_dict = dict(), dict(), dict(), dict(), dict()
        for THRESHOLD_DISTANCE in range(1, max_threshold_distance):
            curr_df = df.loc[(df['label'] == f'{at_most_least}_{THRESHOLD_DISTANCE}_{span}') & (df['split'] == SPLIT)]
            num_examples_dict[THRESHOLD_DISTANCE] = curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0]
            if curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0] > MIN_EXAMPLES_CNT:
                exp_layer_dict[THRESHOLD_DISTANCE], first_negative_delta_dict[THRESHOLD_DISTANCE], var_layer_dict[
                    THRESHOLD_DISTANCE], best_layer_dict[THRESHOLD_DISTANCE] = calc_expected_layer(curr_df)
        return exp_layer_dict, first_negative_delta_dict, var_layer_dict, best_layer_dict, num_examples_dict

def TCE_calculate(df1,df2,max_thr_distance1,max_thr_distance2,allSpans, span1,span2):
    # Total Casual Effect (TCE) of changing from Grammer task whose df is df1 to Grammer task whose df is df2
    exp_layer_dict1, exp_layer_dict2, span_prob_dict1, span_prob_dict2 = get_exp_prob(df1,df2,max_thr_distance1,max_thr_distance2, allSpans, span1, span2)
    total_exp1 = sum([exp_layer_dict1[k] * span_prob_dict1[k] for k in exp_layer_dict1.keys() if k in exp_layer_dict2.keys()]) # according to the total expectation formula
    total_exp2 = sum([exp_layer_dict2[k] * span_prob_dict2[k] for k in exp_layer_dict2.keys() if k in exp_layer_dict1.keys()]) # according to the total expectation formula
    return total_exp2-total_exp1

def CDE_calculate(df1,df2,max_thr_distance1,max_thr_distance2,allSpans, span1,span2):
    # Controlled Direct Effect (CDE) of changing from Grammer task whose df is df1 to Grammer task whose df is df2
    exp_layer_dict1, exp_layer_dict2, span_prob_dict1, span_prob_dict2 = get_exp_prob(df1, df2, max_thr_distance1, max_thr_distance2, allSpans, span1, span2)
    return {k : (exp_layer_dict2[k]- exp_layer_dict1[k]) for k in exp_layer_dict2.keys() if k in exp_layer_dict1.keys()}

def NDE_calculate(df1,df2,max_thr_distance1,max_thr_distance2,allSpans, span1,span2):
    # Natural Direct Effect (NDE) of changing from Grammer task whose df is df1 to Grammer task whose df is df2
    exp_layer_dict1, exp_layer_dict2, span_prob_dict1, span_prob_dict2 = get_exp_prob(df1, df2, max_thr_distance1, max_thr_distance2, allSpans, span1, span2)
    diff = {k : (exp_layer_dict2[k]- exp_layer_dict1[k]) for k in exp_layer_dict2.keys() if k in exp_layer_dict1.keys()}
    return sum([span_prob_dict1[k]*diff[k] for k in diff.keys()])

def NIE_calculate(df1,df2,max_thr_distance1,max_thr_distance2,allSpans, span1,span2):
    # Natural Direct Effect (NDE) of changing from Grammer task whose df is df1 to Grammer task whose df is df2
    exp_layer_dict1, exp_layer_dict2, span_prob_dict1, span_prob_dict2 = get_exp_prob(df1, df2, max_thr_distance1,max_thr_distance2, allSpans, span1, span2)
    diff = {k: (span_prob_dict2[k] - span_prob_dict1[k]) for k in exp_layer_dict2.keys() if k in exp_layer_dict1.keys()}
    return sum([exp_layer_dict1[k] * diff[k] for k in diff.keys()])

def all_effects(df1,df2,max_thr_distance1,max_thr_distance2, allSpans=False, span1=True, span2=True):
    # span1/2 = that we check the span_distance parameter, span1_length or span1_span2_length for df1,df2 respectively
    # returns TCE, CDE. NDE and NIE
    TCE = TCE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=True, span1=span1, span2=span2)
    CDE = CDE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    NDE = NDE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    NIE = NIE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    return TCE, CDE, NDE, NIE

def seperate_CDE(df,allSpans, span):
    exp_layer_dict1, span_probability1 = TCE_helper(df, MAX_ALL_THRESHOLD_DISTANCE, allSpans=allSpans, span=span)

def get_exp_and_best_layer_dict(df, max_threshold_distance, span = SPAN1_SPAN2_DIST, at_most_least = AT_MOST):
    # span = span type: span1 length (if only span1), distance between span1 and span 2 or the total length of span1
    #           to span2 (or vice versa). can be SPAN1_LEN, SPAN1_SPAN2_LEN or SPAN1_SPAN2_DIST
    # at_most_least = whether the close threshold (<=thr) or the far threshold (>=thr)
    exp_layer_dict , var_layer_dict, first_negative_delta_dict, best_layer_dict, num_examples_dict= dict(), dict(), dict(), dict(), dict()
    for THRESHOLD_DISTANCE in range(1, max_threshold_distance):
        curr_df = df.loc[(df['label'] == f'{at_most_least}_{THRESHOLD_DISTANCE}_{span}') & (df['split'] == SPLIT)]
        num_examples_dict[THRESHOLD_DISTANCE] = curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0]
        if curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0] > MIN_EXAMPLES_CNT:
            exp_layer_dict[THRESHOLD_DISTANCE], first_negative_delta_dict[THRESHOLD_DISTANCE], var_layer_dict[THRESHOLD_DISTANCE], best_layer_dict[THRESHOLD_DISTANCE] = calc_expected_layer(curr_df)
    return exp_layer_dict, first_negative_delta_dict, var_layer_dict,  best_layer_dict, num_examples_dict


def get_all_TCE_CDE_NIE_NDE(df_compare_to,span_compare_to, str_compare_to,max_thr_compare_to, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, df_exp_layer):
    TCE, CDE, NDE, NIE, exp_layer_diff = dict(), dict(), dict(), dict(), dict()

    if str_compare_to != "non-terminals":
        a = str_compare_to + ' to non-terminals'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_nonterminals, max_thr_compare_to,MAX_NONTERMINAL_THRESHOLD_DISTANCE, allSpans=False, span1=span_compare_to,span2=SPAN1_LEN)
        exp_layer_diff[a] = df_exp_layer['non-terminals'] - df_exp_layer[str_compare_to]
    if str_compare_to != "dependencies":
        a = str_compare_to + ' to dependencies'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_dep, max_thr_compare_to,MAX_DEP_THRESHOLD_DISTANCE, allSpans=False,span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['dependencies'] - df_exp_layer[str_compare_to]
    if str_compare_to != "NER":
        a = str_compare_to + ' to NER'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_ner, max_thr_compare_to, MAX_NER_THRESHOLD_DISTANCE, allSpans=False, span1=span_compare_to, span2=SPAN1_LEN)
        exp_layer_diff[a] = df_exp_layer['NER'] - df_exp_layer[str_compare_to]
    if str_compare_to != "SRL":
        a = str_compare_to + ' to SRL'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_srl, max_thr_compare_to, MAX_SRL_THRESHOLD_DISTANCE, allSpans=False, span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['SRL'] - df_exp_layer[str_compare_to]
    if "co-reference" not in str_compare_to:
        a = str_compare_to + ' to co-reference'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_coref, max_thr_compare_to, MAX_COREF_OLD_THRESHOLD_DISTANCE,allSpans=False, span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['co-reference'] - df_exp_layer[str_compare_to]
    if str_compare_to != "relations":
        a = str_compare_to + ' to relations'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_rel, max_thr_compare_to, MAX_REL_THRESHOLD_DISTANCE ,allSpans=False, span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['relations'] - df_exp_layer[str_compare_to]
    if str_compare_to != "SPR":
        a = str_compare_to + ' to SPR'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_spr, max_thr_compare_to, MAX_SPR_THRESHOLD_DISTANCE, allSpans=False, span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['SPR'] - df_exp_layer[str_compare_to]
    return TCE, CDE, NDE, NIE, exp_layer_diff

def biggest_TCE_NDE_diff(TCE_all,NDE_all, NIE_all):
    def calc_change(TCE, NDE):
        increase_decrease = 'decrease' if abs(TCE) > abs(NDE) else 'increase'
        change_difficulty_balance = True if ((TCE > 0 and NDE < 0) or (TCE < 0 and NDE > 0)) else False
        return {'incr/decr':increase_decrease , 'change(%)':abs((TCE - NDE) / TCE)*100, 'change(abs val)':abs(TCE - NDE), 'change difficulty balance':change_difficulty_balance}

    change_dict = {k: (calc_change(TCE_all[ref_task][k], NDE_all[ref_task][k])) for ref_task in NDE_all.keys() for k in NDE_all[ref_task].keys()}
    change_perc_dict = {k: change_dict[k]['change(%)'] for k in change_dict.keys()}
    largest_change = nlargest(6, change_dict, key=change_perc_dict.get)
    return {elem : change_dict[elem] for elem in largest_change}

def impose_max_min(span_exp_layer):
    # returns diffs, a dict whose keys are of structure "k to j" where k and j are grammatical tasks, and the values are lists where each time we subtract k's expected layer from j's, when the first elem of the list j's exp layer is the max of all the spans and k's is the min and the second elem vice versa
    diffs = dict()
    for k in span_exp_layer.keys():
        for j in span_exp_layer.keys():
            if k + ' minus ' + j not in diffs.keys() and j != k:
                k_vals = span_exp_layer[k].values()
                j_vals = span_exp_layer[j].values()
                diffs[j + ' minus ' + k] = {'max minus min' : max(j_vals) - min(k_vals), 'min minus max' : min(j_vals) - max(k_vals)}
    return diffs


def main(args):
    a = DataAll()
    b = Result(a.coreference, MAX_COREF_OLD_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    print('end')



if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)