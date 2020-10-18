import sys, os, re, json
from importlib import reload
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import jiant.probing.analysis as analysis
reload(analysis)
from heapq import nlargest
import seaborn as sns
from jiant import *

tasks = analysis.TASKS
exp_types = analysis.EXP_TYPES
palette = analysis.EXP_PALETTE

task_sort_key = analysis.task_sort_key
exp_type_sort_key = analysis.exp_type_sort_key

SPAN1_LEN = 'span1_len'
SPAN1_SPAN2_LEN = 'span1_span2_len'
SPAN1_SPAN2_DIST = 'span1_span2_dist'
AT_LEAST = "at_least"
AT_MOST = "at_most"

SPLIT = 'val'
MAX_COREF_OLD_THRESHOLD_DISTANCE = 66
MAX_COREF_NEW_THRESHOLD_DISTANCE = 66
MAX_SPR_THRESHOLD_DISTANCE = 24 #changes from the original Edgeprobe_Aggregate_Analysis.py
MAX_SRL_THRESHOLD_DISTANCE = 22
MAX_NER_THRESHOLD_DISTANCE = 9
MAX_NONTERMINAL_THRESHOLD_DISTANCE = 55
MAX_DEP_THRESHOLD_DISTANCE = 30
MAX_ALL_THRESHOLD_DISTANCE = min(MAX_COREF_OLD_THRESHOLD_DISTANCE,MAX_COREF_NEW_THRESHOLD_DISTANCE,MAX_SPR_THRESHOLD_DISTANCE, MAX_SRL_THRESHOLD_DISTANCE, MAX_NER_THRESHOLD_DISTANCE, MAX_NONTERMINAL_THRESHOLD_DISTANCE, MAX_DEP_THRESHOLD_DISTANCE)
BERT_LAYERS=12
MIN_EXAMPLES_CNT = 700
MIN_EXAMPLES_CNT_percent = 0.01 # less then 1% of total samples - ignore
MIN_EXAMPLES_CNT_percent_LEFTOVERS = 0.005
CASUAL_EFFECT_SPAN_SIZE = 3
NER_CASUAL_EFFECT_SPAN_SIZE = CASUAL_EFFECT_SPAN_SIZE

from scipy.special import logsumexp
from scipy.stats import entropy

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

import bokeh
import bokeh.plotting as bp
bp.output_notebook()

import datetime
import socket

ID_COLS = ['run', 'task', 'split']

def agg_stratifier_group(df, stratifier, key_predicate, group_name):
    agg_map = {k:"sum" for k in df.columns if k.endswith("_count")}
    # Use this for short-circuit evaluation, so we don't call key_predicate on invalid keys
    mask = [(s == stratifier and key_predicate(key))
            for s, key in zip(df['stratifier'], df['stratum_key'])]
    sdf = df[mask].groupby(by=ID_COLS).agg(agg_map).reset_index()
    sdf['label'] = group_name
    return sdf

def load_scores_file(filename, tag=None, seed=None, max_threshold_distance=20):
    df = pd.read_csv(filename, sep="\t", header=0)
    df.drop(['Unnamed: 0'], axis='columns', inplace=True)
    # df['task_raw'] = df['task'].copy()
    df['task'] = df['task'].map(analysis.clean_task_name)
    if not "stratifier" in df.columns:
        df["stratifier"] = None
    if not "stratum_key" in df.columns:
        df["stratum_key"] = 0
    # Custom aggregations - Span distances - for every THRESHOLD_DISTANCE between 1 and MAX_SPR_THRESHOLD_DISTANCE split into bigger than THRESHOLD_DISTANCE and smaller than it
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance+1):
        _eg = []
        _eg.append(agg_stratifier_group(df, 'span_distance', lambda x: int(x) <= THRESHOLD_DISTANCE, f'at_most_{THRESHOLD_DISTANCE}_span1_span2_dist'))
        _eg.append(agg_stratifier_group(df, 'span_distance', lambda x: int(x) >= THRESHOLD_DISTANCE, f'at_least_{THRESHOLD_DISTANCE}_span1_span2_dist'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance+1- CASUAL_EFFECT_SPAN_SIZE, CASUAL_EFFECT_SPAN_SIZE):
        _eg = []
        l_bound = THRESHOLD_DISTANCE # lower bound
        h_bound = THRESHOLD_DISTANCE + CASUAL_EFFECT_SPAN_SIZE- 1 # higher bound is minus 1 of the next loewer bound
        _eg.append(agg_stratifier_group(df, 'span_distance', lambda x: l_bound <= int(x) <= h_bound, f'{l_bound}-{h_bound}_span1_span2_dist'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1):
        _eg = []
        _eg.append(agg_stratifier_group(df, 'span1_length', lambda x: int(x) <= THRESHOLD_DISTANCE, f'at_most_{THRESHOLD_DISTANCE}_span1_len'))
        _eg.append(agg_stratifier_group(df, 'span1_length', lambda x: int(x) >= THRESHOLD_DISTANCE, f'at_least_{THRESHOLD_DISTANCE}_span1_len'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance+1- NER_CASUAL_EFFECT_SPAN_SIZE, NER_CASUAL_EFFECT_SPAN_SIZE):
        _eg = []
        l_bound = THRESHOLD_DISTANCE  # lower bound
        h_bound = THRESHOLD_DISTANCE + NER_CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
        _eg.append(agg_stratifier_group(df, 'span1_length', lambda x: l_bound <= int(x) <= h_bound, f'{l_bound}-{h_bound}_span1_len'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1):
        _eg = []
        _eg.append(agg_stratifier_group(df, 'span1_span2_length', lambda x: int(x) <= THRESHOLD_DISTANCE, f'at_most_{THRESHOLD_DISTANCE}_span1_span2_len'))
        _eg.append(agg_stratifier_group(df, 'span1_span2_length', lambda x: int(x) >= THRESHOLD_DISTANCE, f'at_least_{THRESHOLD_DISTANCE}_span1_span2_len'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)
    for THRESHOLD_DISTANCE in range(0, max_threshold_distance+1- NER_CASUAL_EFFECT_SPAN_SIZE, NER_CASUAL_EFFECT_SPAN_SIZE):
        _eg = []
        l_bound = THRESHOLD_DISTANCE  # lower bound
        h_bound = THRESHOLD_DISTANCE + NER_CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
        _eg.append(agg_stratifier_group(df, 'span1_span2_length', lambda x: l_bound <= int(x) <= h_bound, f'{l_bound}-{h_bound}_span1_span2_len'))
        df = pd.concat([df] + _eg, ignore_index=True, sort=False)


    df.insert(0, "exp_name", df['run'].map(lambda p: os.path.basename(os.path.dirname(p.strip("/")))))
    df.insert(1, "exp_type", df['exp_name'].map(analysis.get_exp_type))
    df.insert(1, "layer_num", df['exp_name'].map(analysis.get_layer_num))
    df.insert(11,"total_count", df['fn_count'] + df['fp_count'] + df['tn_count'] + df['tp_count'])
    if tag is not None:
        df.insert(0, "tag", tag)
    df.insert(1, "seed", seed)
    return df

def _format_display_col(exp_type, layer_num, tag):
    ret = exp_type
    if layer_num:
        ret += f"-{layer_num}"
    if tag:
        ret += f" ({tag})"
    return ret

def get_data(file_name, max_threshold_distance):
    score_files = [("mix", "/cs/labs/oabend/lovodkin93/jiant_rep/jiant-ep_frozen_20190723/probing/" + file_name)] #changes from the original Edgeprobe_Aggregate_Analysis.py
    dfs = []
    for tag, score_file in score_files:
        df = load_scores_file(score_file, tag=tag,max_threshold_distance=max_threshold_distance)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    df['display_col'] = list(map(_format_display_col, df.exp_type, df.layer_num, df.tag))
    analysis.score_from_confusion_matrix(df)

    def _get_final_score(row):
        return row['f1_score'], row['f1_errn95']

    df['score'], df['score_errn95'] = zip(*(_get_final_score(row) for i, row in df.iterrows()))
    return df

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


def get_all_TCE_CDE_NIE_NDE(df_compare_to,span_compare_to, str_compare_to,max_thr_compare_to, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, df_exp_layer):
    TCE, CDE, NDE, NIE, exp_layer_diff = dict(), dict(), dict(), dict(), dict()

    if str_compare_to != "non-terminals":
        a = str_compare_to + ' to non-terminals'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_nonterminals, max_thr_compare_to,MAX_NONTERMINAL_THRESHOLD_DISTANCE, allSpans=False, span1=span_compare_to,span2=SPAN1_LEN)
        exp_layer_diff[a] = df_exp_layer['non-terminals'] - df_exp_layer[str_compare_to]
    if str_compare_to != "Universal Dependencies":
        a = str_compare_to + ' to Universal Dependencies'
        TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(df_compare_to, df_dep, max_thr_compare_to,MAX_DEP_THRESHOLD_DISTANCE, allSpans=False,span1=span_compare_to, span2=SPAN1_SPAN2_DIST)
        exp_layer_diff[a] = df_exp_layer['Universal Dependencies'] - df_exp_layer[str_compare_to]
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
    largest_change = nlargest(3, change_dict, key=change_perc_dict.get)
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


def plot(y1,y2, label1, label2 ,max_threshold_1,max_threshold_2,xlabel,ylabel,title,withLegend,withY2,subplt_orient=111):
    # import seaborn as sns
    # y1_df = pd.DataFrame({"max_len": list(y1.keys()), "expected_layer": list(y1.values())})
    # y2_df = pd.DataFrame({"max_len": list(y2.keys()), "expected_layer": list(y2.values())})
    # sns.lineplot(data=y1_df, x="max_len", y="expected_layer", markers=True, dashes=False)
    # frames = [y1_df, y2_df]
    # df_keys = pd.merge(y1_df, y2_df, on='max_len')


    # plots the graph whose data is in y1 and if withY2==True than in y2 also. If withLegend==True, add legend
    plt.rcParams.update({'font.size': 18})
    if subplt_orient==111:
        plt.rcParams.update({'font.size': 22})
    plt.subplots_adjust(hspace=0.3)
    if  subplt_orient % 10 == 1: # the first subplot
        plt.suptitle(title)
    plt.subplot(subplt_orient)
    max_x_value = min(max_threshold_1, max_threshold_2, len(y1)+1, len(y2)+1) if (withY2 == True) else min(max_threshold_1, len(y1)+1)
    x_axis = list(range(1, max_x_value))
    plt.figure(1)
    y_axis_1 = [y1[i] for i in x_axis]
    plt.plot(x_axis, y_axis_1, '-ok', label=label1)
    if (withY2 == True):
        y_axis_2 = [y2[i] for i in x_axis]
        plt.plot(x_axis, y_axis_2, ':^r', label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(withLegend == True):
        plt.legend()
    if ((subplt_orient // 100) % 100) * ((subplt_orient // 10) % 10) == subplt_orient % 10: # last graph of the subplot, for example subplt_orient=224, so 2*2=4
        plt.show()


def plots_2_tasks(coref, coref_inSent, spr, srl, ner, nonterminals, ylabel=""):
    title = ylabel + ' as a func. of maximal distance '
   # non-terminals vs. co-reference
    plot(coref, nonterminals, 'co-reference', 'non-terminals',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_NONTERMINAL_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,221)
    # NER vs. co-reference
    plot(coref, ner, 'co-reference', 'NER',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_NER_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,222)
    # srl vs. co-reference
    plot(coref, srl, 'co-reference', 'SRL',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_SRL_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,223)
    # spr vs. co-reference
    plot(coref, spr, 'co-reference', 'SPR',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_SPR_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,224)
    # non-terminals vs. in-sentence co-reference
    plot(coref_inSent, nonterminals, 'co-reference', 'non-terminals',
         MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_NONTERMINAL_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,221)
    # NER vs. in-sentence co-reference
    plot(coref_inSent, ner, 'co-reference', 'NER',
         MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_NER_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,222)
    # srl vs. in-sentence co-reference
    plot(coref_inSent, srl, 'co-reference', 'SRL',
        MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_SRL_THRESHOLD_DISTANCE,
        'Threshold distance', ylabel, title, True, True,223)
    # spr vs. in-sentence co-reference
    plot(coref_inSent, spr, 'co-reference', 'SPR',
         MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_SPR_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,224)
   # co-reference vs. in-sentence co-reference
    plot(coref, coref_inSent, 'co-reference', 'in-sentence co-reference',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_COREF_NEW_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,111)


def plots_1_task(coref, coref_inSent, spr, srl, ner, nonterminals, ylabel=""):
    # in-sentence co-reference
    plot(coref_inSent, [], 'in-sentence co-reference', '', MAX_COREF_NEW_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of maximal distance - New Coreference task', False, False)
    #co-reference
    plot(coref, [], 'co-reference', '', MAX_COREF_OLD_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of maximal distance - Old Coreference task', False, False)
    # SPR
    plot(spr, [], 'SPR', '', MAX_SPR_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of maximal distance - SPR task', False, False)
    # SRL
    plot(srl, [], 'SRL', '', MAX_SRL_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of maximal distance - SRL task', False, False)
    # NER
    plot(ner, [], 'NER', '', MAX_NER_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of minimal distance - NER task', False, False)
    # non-terminals
    plot(nonterminals, [], 'non-terminals', '', MAX_NONTERMINAL_THRESHOLD_DISTANCE, -1, 'Threshold distance',
         ylabel, ylabel + ' as a func. of minimal distance - non-terminals task', False, False)

def plot_span_expected_layer(span_exp_layer, x_label, y_label, xlim , barGraphToo=True):

    def relevant_sub_df(span_exp_layer, task):
        span_exp_layer_df = pd.DataFrame(span_exp_layer)
        span_exp_layer_df[x_label] = span_exp_layer_df.index
        sub_df = pd.DataFrame(span_exp_layer_df[[x_label, task]])
        sub_df['task'] = task
        sub_df = sub_df.rename(columns={task: y_label})
        return sub_df

    def get_new_df(span_exp_layer):
        new_df = pd.DataFrame(columns=[x_label, y_label, 'task'])
        span_exp_layer_df = pd.DataFrame(span_exp_layer)
        for task in span_exp_layer_df.columns:
            new_df = new_df.append(relevant_sub_df(span_exp_layer_df, task))
        return new_df

    new_df = get_new_df(span_exp_layer)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )

    lnp = sns.lineplot(x=x_label, y=y_label, data=new_df, hue="task",
                 style="task", palette="hot", dashes=False,
                 markers=["o", "<", ">", "*", "d", "X" ], legend="brief", )
    axes = lnp.axes
    axes.set_xlim(-0.1, xlim + 0.1)

    if barGraphToo:
        plt.figure(figsize=(7, 9))
        sns.set(style='darkgrid', )
        sns.boxplot(x='task', y=y_label, data=new_df, whis=100, width=0)

def plot_all_CDE(CDE, justOld=False, justNew=False):

    def plot_cde(cde, title, subplt_orient=111):
        plt.rcParams.update({'font.size': 18})
        plt.subplots_adjust(hspace=0.4)
        plt.subplot(subplt_orient)
        x_axis = list(cde.keys())
        plt.figure(1)
        y_axis = [cde[i] for i in x_axis]
        plt.bar(x_axis, y_axis)
        plt.xlabel('Spans')
        plt.ylabel('CDE')
        plt.title("CDE of " + title)

    def find_divider(num):
        for i in list(range(2,num)):
            if num % i == 0:
                return i
        return 1

    cde = dict()

    if justOld:
        cde = {key: CDE[key] for key in CDE.keys() if "sentence" not in key}
    elif justNew:
        cde = {key: CDE[key] for key in CDE.keys() if "sentence" in key}
    else:
        cde = CDE

    initial_subplt_orient = 100 * (find_divider(len(cde))) + 10 * (len(cde) // find_divider(len(cde)))
    for idx, k in enumerate(cde.keys()):
        subplt_orient = initial_subplt_orient + idx + 1
        plot_cde(cde[k], '-' + k + '-', subplt_orient)
    plt.show()


def plot_TCE_NDE_NIE(TCE, NDE, NIE, exp_layer_diff, specific_tasks=None, noTCE=False, noNDE=False, noNIE=False, noExpLayerDiff=False):

    def get_relevant_df(dict, name):
        df = pd.DataFrame(dict)
        df['result'] = df[df.columns[0:]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
        )
        df['tasks'] = df.index
        df['value'] = name
        df = df[['value', 'tasks', 'result']]
        return df

    def clean_df(total_df):
        cleaned_df = total_df.loc[total_df['tasks'].isin(specific_tasks)]
        cleaned_df = cleaned_df.loc[cleaned_df['value'] != 'TCE'] if noTCE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['value'] != 'NDE'] if noNDE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['value'] != 'NIE'] if noNIE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['value'] != 'Difference in Expected Layer'] if noExpLayerDiff else cleaned_df
        return cleaned_df


    def get_total_df():
        TCE_df = get_relevant_df(TCE, 'TCE')
        NDE_df = get_relevant_df(NDE, 'NDE')
        NIE_df = get_relevant_df(NIE, 'NIE')
        exp_layer_diff_df = get_relevant_df(exp_layer_diff, 'Difference in Expected Layer')
        total_df = pd.concat([TCE_df, NDE_df, NIE_df, exp_layer_diff_df])
        total_df = clean_df(total_df)
        return total_df

    total_df = get_total_df()
    # total_df = total_df.replace('non-terminals to SRL', 'E(SRL)-E(non-terminals)\n(for NDE - non-terminals span distribution)')
    # total_df = total_df.replace('co-reference to NER', 'E(co-reference)-E(NER)\n(for NDE - co-reference span distribution)')
    # total_df = total_df.replace('SPR to co-reference', 'E(SPR)-E(co-reference)\n(for NDE - SPR span distribution)')
    # total_df['result'] = pd.DataFrame.abs(pd.to_numeric(total_df['result']))
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    lnp = sns.barplot(x='tasks', y='result', data=total_df, hue="value", palette="hot")

def plot_diffs_max_min(diffs_max_min):
    import math
    def relevant_sub_df(diffs_max_min, task):
        diffs_max_min_df = pd.DataFrame(diffs_max_min)
        diffs_max_min_df['task_order'] = diffs_max_min_df.index
        sub_df = pd.DataFrame(diffs_max_min_df[['task_order', task]])
        sub_df['task'] = task
        sub_df = sub_df.rename(columns={task: 'diff between exp layers'})
        sub_df = sub_df.loc[sub_df['diff between exp layers'] == sub_df['diff between exp layers']] #to get rid of None
        return sub_df

    def get_new_df(diffs_max_min):
        new_df = pd.DataFrame(columns=['diff between exp layers', 'task'])
        diffs_max_min_df = pd.DataFrame(diffs_max_min)
        for task in diffs_max_min_df.columns:
            new_df = new_df.append(relevant_sub_df(diffs_max_min_df, task))
        return new_df

    new_df = get_new_df(diffs_max_min)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    sns.barplot(x="task_order", y='diff between exp layers', data=new_df, hue="task",
                palette="hot", )

def plot_sympson_paradox(span_dict, simple_task, complex_task):
    def relevant_sub_df(df, task):
        span_df = pd.DataFrame(span_dict)
        span_df['spans'] = span_df.index
        sub_df = pd.DataFrame(span_df[['spans', task]])
        sub_df['task'] = task
        sub_df = sub_df.rename(columns={task: 'Expected Layer'})
        sub_df = sub_df.loc[sub_df['Expected Layer'] == sub_df['Expected Layer']]
        return sub_df

    def get_new_df(span_dict):
        new_df = pd.DataFrame(columns=['Expected Layer', 'task'])
        span_df = pd.DataFrame(span_dict)
        for task in span_df.columns:
            new_df = new_df.append(relevant_sub_df(span_df, task))
        return new_df
    new_df = get_new_df(span_dict)
    new_df = new_df.loc[(new_df['task'] == simple_task) | (new_df['task'] == complex_task)]
    new_df = new_df.append({'task': 'co-reference (span: 0-2)', 'spans': '', 'Expected Layer': span_dict[complex_task]['0-2']}, ignore_index=True)
    new_df = new_df.append({'task': 'NER (span: 9+)', 'spans': '', 'Expected Layer': span_dict[simple_task]['9+']}, ignore_index=True)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    sns.barplot(x="spans", y='Expected Layer', data=new_df, hue="task",
                palette="hot", )
    xlabels = tuple(pd.Series.unique(new_df['spans']))
    plt.xticks(np.arange(len(xlabels)) - 0.2, xlabels, rotation=0, fontsize="10", va="center")

def main(args):
    # dfs
    df_coref = get_data("./scores/scores_old_coref.tsv", MAX_COREF_OLD_THRESHOLD_DISTANCE)
    df_coref_inSent = get_data("./scores/scores_new_coref.tsv", MAX_COREF_NEW_THRESHOLD_DISTANCE)
    df_spr = get_data("./scores/scores_spr1.tsv", MAX_SPR_THRESHOLD_DISTANCE)
    df_srl = get_data("./scores/scores_srl.tsv", MAX_SRL_THRESHOLD_DISTANCE)
    df_ner = get_data("./scores/scores_ner.tsv", MAX_NER_THRESHOLD_DISTANCE)
    df_nonterminals = get_data("./scores/scores_nonterminal.tsv", MAX_NONTERMINAL_THRESHOLD_DISTANCE)
    df_dep = get_data("./scores/scores_dep.tsv", MAX_DEP_THRESHOLD_DISTANCE)

    ############################## span1_span2_distance #############################

    #expected layers, first neative delta, best layers and number of examples:
    coref_exp_layer_dict, coref_first_negative_delta_dict, coref_var_layer_dict, coref_best_layer_dict, coref_num_examples  = get_exp_and_best_layer_dict(df_coref, MAX_COREF_OLD_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    coref_inSent_exp_layer_dict, coref_inSent_first_negative_delta_dict, coref_inSent_var_layer_dict, coref_inSent_best_layer_dict, coref_inSent_num_examples = get_exp_and_best_layer_dict(df_coref_inSent, MAX_COREF_NEW_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    spr_exp_layer_dict, spr_first_negative_delta_dict, spr_var_layer_dict, spr_best_layer_dict, spr_num_examples = get_exp_and_best_layer_dict(df_spr, MAX_SPR_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    srl_exp_layer_dict, srl_first_negative_delta_dict, srl_var_layer_dict, srl_best_layer_dict, srl_num_examples = get_exp_and_best_layer_dict(df_srl, MAX_SRL_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    ner_exp_layer_dict, ner_first_negative_delta_dict, ner_var_layer_dict, ner_best_layer_dict, ner_num_examples = get_exp_and_best_layer_dict(df_ner, MAX_NER_THRESHOLD_DISTANCE, span=SPAN1_LEN , at_most_least=AT_MOST)
    nonterm_exp_layer_dict, nonterm_first_negative_delta_dict, nonterm_var_layer_dict, nonterm_best_layer_dict, nonterm_num_examples = get_exp_and_best_layer_dict(df_nonterminals, MAX_NONTERMINAL_THRESHOLD_DISTANCE , span=SPAN1_LEN, at_most_least=AT_MOST)
    dep_exp_layer_dict, dep_first_negative_delta_dict, dep_var_layer_dict, dep_best_layer_dict, dep_num_examples = get_exp_and_best_layer_dict(df_dep, MAX_DEP_THRESHOLD_DISTANCE, span=SPAN1_SPAN2_DIST, at_most_least=AT_MOST)

    #getting the same value for the expected layer as in the article (3.62 and almost 4.29 and 2.713 for the NER task and 1.936 for nonterminals)
    exp_layer_all = dict()
    #tmp = df_coref.loc[(df_coref['label'] == '0') & (df_coref['split'] == 'val')]
    tmp = df_coref.loc[(df_coref['label'] == '_micro_avg_') & (df_coref['split'] == 'val')]
    exp_layer_all['co-reference'], _, _, _ = calc_expected_layer(tmp)
    #tmp = df_coref_inSent.loc[(df_coref_inSent['label'] == '0') & (df_coref_inSent['split'] == 'val')]
    #tmp = df_coref_inSent.loc[(df_coref_inSent['label'] == '_micro_avg_') & (df_coref_inSent['split'] == 'val')]
    #exp_layer_all['in-sentence co-reference'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_srl.loc[(df_srl['label'] == '_micro_avg_') & (df_srl['split'] == 'val')]
    exp_layer_all['SRL'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_spr.loc[(df_spr['label'] == '_micro_avg_') & (df_spr['split'] == 'val')]
    exp_layer_all['SPR'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_ner.loc[(df_ner['label'] == '_micro_avg_') & (df_ner['split'] == 'val')]
    exp_layer_all['NER'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_nonterminals.loc[(df_nonterminals['label'] == '_micro_avg_') & (df_nonterminals['split'] == 'val')]
    exp_layer_all['non-terminals'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_dep.loc[(df_dep['label'] == '_micro_avg_') & (df_dep['split'] == 'val')]
    exp_layer_all['Universal Dependencies'], _, _, _ = calc_expected_layer(tmp)

################################################# EXP LAYER PER SPAN & SPAN DISTRIBUTION #############################################################
    span_exp_layer, span_prob = dict(), dict()
    span_exp_layer['co-reference'], span_prob['co-reference'] = TCE_helper(df_coref, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    #span_exp_layer['in-sentence co-reference'], span_prob['in-sentence co-reference'] = TCE_helper(df_coref_inSent, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['non-terminals'], span_prob['non-terminals'] = TCE_helper(df_nonterminals, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_LEN)
    span_exp_layer['Universal Dependencies'], span_prob['Universal Dependencies'] = TCE_helper(df_dep, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['NER'], span_prob['NER'] = TCE_helper(df_ner, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_LEN)
    span_exp_layer['SRL'], span_prob['SRL'] = TCE_helper(df_srl, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['spr'], span_prob['spr'] = TCE_helper(df_spr, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)

    plot_span_expected_layer(span_exp_layer,"spans", "expected layer", xlim=3, barGraphToo=True)
    plot_span_expected_layer(span_prob, "spans", "span probability", xlim=3, barGraphToo=False)

    diffs_max_min = impose_max_min(span_exp_layer)
    plot_diffs_max_min(diffs_max_min)
    plot_sympson_paradox(span_exp_layer, 'NER', 'co-reference')
############################################################# ALL COMBOS ##################################################################

    TCE_all, CDE_all, NDE_all, NIE_all, exp_layer_diff_all = dict(), dict(), dict(), dict(), dict()

    a = 'non-terminals'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_nonterminals, SPAN1_LEN, "non-terminals", MAX_NONTERMINAL_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)
    a='NER'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_ner, SPAN1_LEN, "NER", MAX_NER_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)
    a='SRL'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_srl, SPAN1_SPAN2_DIST, "SRL", MAX_SRL_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)
    a='co-reference'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_coref, SPAN1_SPAN2_DIST, "co-reference", MAX_COREF_OLD_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)
    a='SPR'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_spr, SPAN1_SPAN2_DIST, "SPR", MAX_SPR_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)
    a='Universal Dependencies'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_dep, SPAN1_SPAN2_DIST, "Universal Dependencies", MAX_DEP_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_spr, exp_layer_all)

    biggest_TCE_NDE_difference = biggest_TCE_NDE_diff(TCE_all, NDE_all, NIE_all)
    biggest_exp_layer_NDE_difference = biggest_TCE_NDE_diff(exp_layer_diff_all, NDE_all, NIE_all)

    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_TCE_NDE_difference, noTCE=False, noNDE=False, noNIE=True, noExpLayerDiff=True)
    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_exp_layer_NDE_difference, noTCE=True, noNDE=False, noNIE=True, noExpLayerDiff=False)

    plot_all_CDE(CDE_all['non-terminals'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['Universal Dependencies'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['NER'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['SRL'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['co-reference'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['SPR'], justOld=False, justNew=False)

    ###########################################################################################################################################
    # # for comparison with the TCE values:
    # a = df_coref.loc[(df_coref['label'] == '_macro_avg_') & (df_coref['split'] == SPLIT)]
    # exp_coref, _, _, _ = calc_expected_layer(a)
    # b = df_srl.loc[(df_srl['label'] == '_macro_avg_') & (df_srl['split'] == SPLIT)]
    # exp_srl, _, _, _ = calc_expected_layer(b)
##################################################################################
    exp_layer = dict()
    exp_layer['non-terminals'] = nonterm_exp_layer_dict
    exp_layer['Universal Dependencies'] = dep_exp_layer_dict
    exp_layer['NER'] = ner_exp_layer_dict
    exp_layer['SRL'] = srl_exp_layer_dict
    exp_layer['co-reference'] = coref_exp_layer_dict
    exp_layer['SPR'] = spr_exp_layer_dict
    plot_span_expected_layer(exp_layer, "threshold", "expected layer", xlim=MAX_SPR_THRESHOLD_DISTANCE, barGraphToo=False)
################################################## PLOTS #############################################################
    plots_2_tasks(coref_exp_layer_dict, coref_inSent_exp_layer_dict, spr_exp_layer_dict, srl_exp_layer_dict, ner_exp_layer_dict,nonterm_exp_layer_dict, ylabel="Expected Layer")
    # plots_2_tasks(coref_var_layer_dict, coref_inSent_var_layer_dict, spr_var_layer_dict, srl_var_layer_dict, ner_var_layer_dict,nonterm_var_layer_dict, ylabel="Variance of Layer")
    # plots_2_tasks(coref_best_layer_dict, coref_inSent_best_layer_dict, spr_best_layer_dict, srl_best_layer_dict,ner_best_layer_dict,nonterm_best_layer_dict, ylabel="Best Layer")
    #
    # plots_1_task(coref_best_layer_dict, coref_inSent_best_layer_dict, spr_best_layer_dict, srl_best_layer_dict, ner_best_layer_dict,nonterm_best_layer_dict, ylabel="Best Layer")
    # plots_1_task(coref_num_examples, coref_inSent_num_examples, spr_num_examples, srl_num_examples, ner_num_examples, nonterm_num_examples, ylabel="Number of examples")
    # plots_1_task(coref_first_negative_delta_dict, coref_inSent_first_negative_delta_dict, spr_first_negative_delta_dict, srl_first_negative_delta_dict, ner_first_negative_delta_dict,nonterm_first_negative_delta_dict, "layer where starts negative delta")



if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)