import sys, os, re, json
import itertools
import collections
import numpy as np
import pandas as pd
from heapq import nlargest
from jiant import *
from scipy.special import logsumexp
import matplotlib.pyplot as plt

SPAN1_LEN = 'span1_len'
SPAN1_SPAN2_LEN = 'span1_span2_len'
SPAN1_SPAN2_DIST = 'span1_span2_dist'
TWO_SPANS_SPAN = SPAN1_SPAN2_LEN
ONE_SPAN_SPAN = SPAN1_LEN
AT_LEAST = "at_least"
AT_MOST = "at_most"

SPLIT = 'val'
MAX_COREF_OLD_THRESHOLD_DISTANCE = 69 #66
MAX_COREF_NEW_THRESHOLD_DISTANCE = 69 #66
MAX_SPR_THRESHOLD_DISTANCE = 28 #24
MAX_SRL_THRESHOLD_DISTANCE = 46 #22
MAX_NER_THRESHOLD_DISTANCE = 9 #9
MAX_NONTERMINAL_THRESHOLD_DISTANCE = 60 #55
MAX_DEP_THRESHOLD_DISTANCE = 36 #30
MAX_REL_THRESHOLD_DISTANCE = 20 # 9
MAX_ALL_THRESHOLD_DISTANCE = min(MAX_COREF_OLD_THRESHOLD_DISTANCE,MAX_COREF_NEW_THRESHOLD_DISTANCE,MAX_SPR_THRESHOLD_DISTANCE, MAX_SRL_THRESHOLD_DISTANCE, MAX_NER_THRESHOLD_DISTANCE, MAX_NONTERMINAL_THRESHOLD_DISTANCE, MAX_DEP_THRESHOLD_DISTANCE, MAX_REL_THRESHOLD_DISTANCE)
BERT_LAYERS=12
MIN_EXAMPLES_CNT = 700
MIN_EXAMPLES_CNT_percent = 0.01 # less then 1% of total samples - ignore
MIN_EXAMPLES_CNT_percent_LEFTOVERS = 0.004
CASUAL_EFFECT_SPAN_SIZE = 3

ID_COLS = ['run', 'task', 'split']

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def calc_expected_layer(df):
    # returns the expected layer and the num of layers where there's a negative delta for the first time
    if (len(df) == 0):
       return 0, 0, 0, 0
    f1_scores = df[['layer_num', 'f1_score']]
    numerator_X , numerator_X_2, denominator, best_num_layer, first_neg_delta= 0,0,0,0,-1  # EX - of expected layer
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
        numerator_X_2 = numerator_X_2 + ((i ** 2) * delta)
        denominator = denominator + delta
    if isZero:
        exp_layer, var_layer = 0, 0
    elif denominator == 0:
        exp_layer, var_layer = BERT_LAYERS, 0
    else:
        exp_layer = numerator_X / denominator
        var_layer = (numerator_X_2 / denominator) - (exp_layer ** 2)  # varX = EX^2 - (EX)^2
    return exp_layer, first_neg_delta, var_layer, best_num_layer

def TCE_helper(df, max_threshold_distance, allSpans=False, span=TWO_SPANS_SPAN):
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

def all_effects(df1,df2,max_thr_distance1,max_thr_distance2, allSpans=False, span1=TWO_SPANS_SPAN, span2=TWO_SPANS_SPAN):
    # span1/2 = that we check the span_distance parameter, span1_length or span1_span2_length for df1,df2 respectively
    # returns TCE, CDE. NDE and NIE
    TCE = TCE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=True, span1=span1, span2=span2)
    CDE = CDE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    NDE = NDE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    NIE = NIE_calculate(df1, df2, max_thr_distance1, max_thr_distance2, allSpans=allSpans, span1=span1, span2=span2)
    return TCE, CDE, NDE, NIE

def min_span_less_one_percent(df,max_threshold_distance,span):
    _, span_probability_dic = TCE_helper(df, max_threshold_distance, allSpans=True, span=span)
    span_probability_df = pd.DataFrame(list(span_probability_dic.values()))
    # first idx when the span prob < 1% and mul by the casual effect span size to normalize (unless there's no such and them return the maximinum span possible
    if np.any(span_probability_df <= MIN_EXAMPLES_CNT_percent)[0]:
        return (np.argmax([span_probability_df<=MIN_EXAMPLES_CNT_percent])) * CASUAL_EFFECT_SPAN_SIZE
    return (len(span_probability_df) - 1) * CASUAL_EFFECT_SPAN_SIZE

def get_exp_prob(df1,df2,max_threshold_distance1,max_threshold_distance2, allSpans=False, span1=TWO_SPANS_SPAN ,span2=TWO_SPANS_SPAN):
    max_threshold_distance = min(min_span_less_one_percent(df1,max_threshold_distance1,span1), min_span_less_one_percent(df2,max_threshold_distance2,span2))
    exp_layer_dict1, span_probability1 = TCE_helper(df1, max_threshold_distance, allSpans=allSpans,span=span1)
    exp_layer_dict2, span_probability2 = TCE_helper(df2, max_threshold_distance, allSpans=allSpans,span=span2)
    return exp_layer_dict1, exp_layer_dict2, span_probability1, span_probability2