import sys, os, re, json
import itertools
import collections
import numpy as np
import pandas as pd
from heapq import nlargest
from jiant import *
from utils import *

def calc_expected_layer(df):
    # returns the expected layer and the num of layers where there's a negative delta for the first time
    if(len(df)==0):
        return 0,0,0,0
    f1_scores = df[['layer_num', 'f1_score']]
    numerator_X = 0 # EX - of expected layer
    numerator_X_2 = 0 # EX^2
    denominator = 0
    best_num_layer = 0
    first_neg_delta = -1
    best_score = f1_scores.loc[f1_scores['layer_num'] == '0']['f1_score'].values[0]
    isZero = True  # make sure it's not a constant zero

    for i in range(1, BERT_LAYERS + 1):
        prev_score = f1_scores.loc[f1_scores['layer_num'] == str(i - 1)]['f1_score'].values[0]
        curr_score = f1_scores.loc[f1_scores['layer_num'] == str(i)]['f1_score'].values[0]
        # best score
        if (curr_score > best_score):
            best_score = curr_score
            best_num_layer = i
        # expected layer, variance of layer and the first negative delta
        delta = curr_score - prev_score
        if (delta != 0):
            isZero = False
        if (first_neg_delta == -1 and delta < 0):
            first_neg_delta = i
        numerator_X = numerator_X + (i * delta)
        numerator_X_2 = numerator_X_2 + ((i**2) * delta)
        denominator = denominator + delta
    if isZero:
        exp_layer = 0
        var_layer = 0
    elif denominator == 0:
        exp_layer = BERT_LAYERS
        var_layer = 0
    else:
        exp_layer = numerator_X / denominator
        var_layer = (numerator_X_2 / denominator) - (exp_layer**2) # varX = EX^2 - (EX)^2
    return exp_layer, first_neg_delta, var_layer, best_num_layer

def TCE_helper(df, max_threshold_distance, allSpans=False, span=SPAN1_SPAN2_DIST):
    # span = types of span: SPAN1_LEN, SPAN1_SPAN2_LEN, SPAN1_SPAN2_DIST
    # returns the expected layer for each spans of coref_span and their probability
    exp_layer_dict = dict()
    num_examples_dict = dict()
    total_example_num = df.loc[(df['label'] == '_macro_avg_') & (df['split'] == SPLIT)]['total_count'].values[0]
    for MIN_DIST in range(0, max_threshold_distance+1- CASUAL_EFFECT_SPAN_SIZE, CASUAL_EFFECT_SPAN_SIZE):
        l_bound = MIN_DIST  # lower bound
        h_bound = MIN_DIST + CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
        curr_df = df.loc[(df['label'] == f'{l_bound}-{h_bound}_{span}') & (df['split'] == SPLIT)]
        num_examples_dict[f'{l_bound}-{h_bound}'] = 0 if len(curr_df) == 0 else curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0]
        if allSpans: # if include all spans, including w/ small dist
            exp_layer_dict[f'{l_bound}-{h_bound}'], _, _, _ = calc_expected_layer(curr_df)
        elif num_examples_dict[f'{l_bound}-{h_bound}'] / total_example_num > MIN_EXAMPLES_CNT_percent: # (len(curr_df) != 0) and (curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0] > MIN_EXAMPLES_CNT):
            exp_layer_dict[f'{l_bound}-{h_bound}'], _, _, _ = calc_expected_layer(curr_df)
    # the rest
    curr_df = df.loc[(df['label'] == f'{AT_LEAST}_{max_threshold_distance}_{span}') & (df['split'] == SPLIT)]
    num_examples_dict[f'{max_threshold_distance}+'] = 0 if len(curr_df) == 0 else curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0]
    if allSpans: # if include all spans, including w/ small dist
        exp_layer_dict[f'{max_threshold_distance}+'], _, _, _ = calc_expected_layer(curr_df)
    elif num_examples_dict[f'{max_threshold_distance}+'] / total_example_num > MIN_EXAMPLES_CNT_percent_LEFTOVERS: #(len(curr_df) != 0) and (curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0] > MIN_EXAMPLES_CNT):
       exp_layer_dict[f'{max_threshold_distance}+'], _, _, _ = calc_expected_layer(curr_df)
    span_probability = {k : num_examples_dict[k]/total_example_num for k in num_examples_dict.keys()}
    return exp_layer_dict, span_probability

def min_span_less_one_percent(df,max_threshold_distance,span):
    _, span_probability_dic = TCE_helper(df, max_threshold_distance, allSpans=True, span=span)
    span_probability_df = pd.DataFrame(list(span_probability_dic.values()))
    # first idx when the span prob < 1% and mul by the casual effect span size to normalize (unless there's no such and them return the maximinum span possible
    if np.any(span_probability_df <= MIN_EXAMPLES_CNT_percent):
        return (np.argmax(span_probability_df<=MIN_EXAMPLES_CNT_percent)) * CASUAL_EFFECT_SPAN_SIZE
    return (len(span_probability_df) - 1) * CASUAL_EFFECT_SPAN_SIZE

def get_exp_prob(df1,df2,max_threshold_distance1,max_threshold_distance2, allSpans=False, span1=SPAN1_SPAN2_DIST ,span2=SPAN1_SPAN2_DIST):
    max_threshold_distance = min(min_span_less_one_percent(df1,max_threshold_distance1,span1), min_span_less_one_percent(df2,max_threshold_distance2,span2))
    exp_layer_dict1, span_probability1 = TCE_helper(df1, max_threshold_distance, allSpans=allSpans,span=span1)
    exp_layer_dict2, span_probability2 = TCE_helper(df2, max_threshold_distance, allSpans=allSpans,span=span2)
    return exp_layer_dict1, exp_layer_dict2, span_probability1, span_probability2

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
    exp_layer_dict = dict()
    var_layer_dict = dict()
    first_negative_delta_dict = dict()
    best_layer_dict = dict()
    num_examples_dict = dict()
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