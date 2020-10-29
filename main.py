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
    # dfs
    df_coref = dataAll.coreference.data_df
    #df_coref_inSent = Data("./scores/scores_new_coref.tsv", MAX_COREF_NEW_THRESHOLD_DISTANCE, 'in-sentence co-reference').data_df
    df_spr = dataAll.spr.data_df
    df_srl = dataAll.srl.data_df
    df_ner = dataAll.ner.data_df
    df_nonterminals = dataAll.nonterminals.data_df
    df_dep = dataAll.dependencies.data_df
    df_rel = dataAll.relations.data_df

    ############################## span1_span2_distance #############################

    #expected layers, first neative delta, best layers and number of examples:
    coref_exp_layer_dict, coref_first_negative_delta_dict, coref_var_layer_dict, coref_best_layer_dict, coref_num_examples  = get_exp_and_best_layer_dict(df_coref, MAX_COREF_OLD_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    #coref_inSent_exp_layer_dict, coref_inSent_first_negative_delta_dict, coref_inSent_var_layer_dict, coref_inSent_best_layer_dict, coref_inSent_num_examples = get_exp_and_best_layer_dict(df_coref_inSent, MAX_COREF_NEW_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    spr_exp_layer_dict, spr_first_negative_delta_dict, spr_var_layer_dict, spr_best_layer_dict, spr_num_examples = get_exp_and_best_layer_dict(df_spr, MAX_SPR_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    srl_exp_layer_dict, srl_first_negative_delta_dict, srl_var_layer_dict, srl_best_layer_dict, srl_num_examples = get_exp_and_best_layer_dict(df_srl, MAX_SRL_THRESHOLD_DISTANCE, at_most_least=AT_MOST)
    ner_exp_layer_dict, ner_first_negative_delta_dict, ner_var_layer_dict, ner_best_layer_dict, ner_num_examples = get_exp_and_best_layer_dict(df_ner, MAX_NER_THRESHOLD_DISTANCE, span=SPAN1_LEN , at_most_least=AT_MOST)
    nonterm_exp_layer_dict, nonterm_first_negative_delta_dict, nonterm_var_layer_dict, nonterm_best_layer_dict, nonterm_num_examples = get_exp_and_best_layer_dict(df_nonterminals, MAX_NONTERMINAL_THRESHOLD_DISTANCE , span=SPAN1_LEN, at_most_least=AT_MOST)
    dep_exp_layer_dict, dep_first_negative_delta_dict, dep_var_layer_dict, dep_best_layer_dict, dep_num_examples = get_exp_and_best_layer_dict(df_dep, MAX_DEP_THRESHOLD_DISTANCE, span=SPAN1_SPAN2_DIST, at_most_least=AT_MOST)
    rel_exp_layer_dict, rel_first_negative_delta_dict, rel_var_layer_dict, rel_best_layer_dict, rel_num_examples = get_exp_and_best_layer_dict(df_rel, MAX_REL_THRESHOLD_DISTANCE, span=SPAN1_SPAN2_DIST, at_most_least=AT_LEAST)

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
    exp_layer_all['dependencies'], _, _, _ = calc_expected_layer(tmp)
    tmp = df_rel.loc[(df_rel['label'] == '_micro_avg_') & (df_rel['split'] == 'val')]
    exp_layer_all['relations'], _, _, _ = calc_expected_layer(tmp)

################################################# EXP LAYER PER SPAN & SPAN DISTRIBUTION #############################################################
    span_exp_layer, span_prob = dict(), dict()
    #span_exp_layer['in-sentence co-reference'], span_prob['in-sentence co-reference'] = TCE_helper(df_coref_inSent, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['non-terminals'], span_prob['non-terminals'] = TCE_helper(df_nonterminals, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_LEN)
    span_exp_layer['dependencies'], span_prob['dependencies'] = TCE_helper(df_dep, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['NER'], span_prob['NER'] = TCE_helper(df_ner, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_LEN)
    span_exp_layer['SRL'], span_prob['SRL'] = TCE_helper(df_srl, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['co-reference'], span_prob['co-reference'] = TCE_helper(df_coref, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['relations'], span_prob['relations'] = TCE_helper(df_rel, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)
    span_exp_layer['SPR'], span_prob['SPR'] = TCE_helper(df_spr, MAX_ALL_THRESHOLD_DISTANCE, allSpans=True, span=SPAN1_SPAN2_DIST)

    plot_span_expected_layer(span_exp_layer,"spans", "expected layer", xlim=3, barGraphToo=True)
    plot_span_expected_layer(span_prob, "spans", "span probability", xlim=3, barGraphToo=False)

    diffs_max_min = impose_max_min(span_exp_layer)
    plot_diffs_max_min(diffs_max_min)
    plot_sympson_paradox(span_exp_layer, 'NER', 'co-reference')
############################################################# ALL COMBOS ##################################################################

    TCE_all, CDE_all, NDE_all, NIE_all, exp_layer_diff_all = dict(), dict(), dict(), dict(), dict()

    a = 'non-terminals'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_nonterminals, SPAN1_LEN, "non-terminals", MAX_NONTERMINAL_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a='NER'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_ner, SPAN1_LEN, "NER", MAX_NER_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a = 'dependencies'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_dep, SPAN1_SPAN2_DIST, "dependencies", MAX_DEP_THRESHOLD_DISTANCE,df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a='SRL'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_srl, SPAN1_SPAN2_DIST, "SRL", MAX_SRL_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a='co-reference'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_coref, SPAN1_SPAN2_DIST, "co-reference", MAX_COREF_OLD_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a = 'relations'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_rel, SPAN1_SPAN2_DIST, "relations", MAX_REL_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)
    a='SPR'
    TCE_all[a], CDE_all[a], NDE_all[a], NIE_all[a], exp_layer_diff_all[a] = get_all_TCE_CDE_NIE_NDE(df_spr, SPAN1_SPAN2_DIST, "SPR", MAX_SPR_THRESHOLD_DISTANCE, df_nonterminals, df_dep, df_ner, df_srl, df_coref, df_rel, df_spr, exp_layer_all)

    biggest_TCE_NDE_difference = biggest_TCE_NDE_diff(TCE_all, NDE_all, NIE_all)
    biggest_exp_layer_NDE_difference = biggest_TCE_NDE_diff(exp_layer_diff_all, NDE_all, NIE_all)

    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_TCE_NDE_difference, noTCE=False, noNDE=False, noNIE=True, noExpLayerDiff=True)
    plot_TCE_NDE_NIE(TCE_all, NDE_all, NIE_all, exp_layer_diff_all, specific_tasks=biggest_exp_layer_NDE_difference, noTCE=True, noNDE=False, noNIE=True, noExpLayerDiff=False)

    plot_all_CDE(CDE_all['non-terminals'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['dependencies'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['NER'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['SRL'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['co-reference'], justOld=False, justNew=False)
    plot_all_CDE(CDE_all['relations'], justOld=False, justNew=False)
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
    exp_layer['dependencies'] = dep_exp_layer_dict
    exp_layer['NER'] = ner_exp_layer_dict
    exp_layer['SRL'] = srl_exp_layer_dict
    exp_layer['co-reference'] = coref_exp_layer_dict
    exp_layer['relations'] = rel_exp_layer_dict
    exp_layer['SPR'] = spr_exp_layer_dict
    plot_span_expected_layer(exp_layer, "threshold", "expected layer", xlim=MAX_SPR_THRESHOLD_DISTANCE, barGraphToo=False)
################################################## PLOTS #############################################################
    plots_2_tasks(coref_exp_layer_dict, spr_exp_layer_dict, srl_exp_layer_dict, ner_exp_layer_dict,nonterm_exp_layer_dict, ylabel="Expected Layer")
    # plots_2_tasks(coref_var_layer_dict, coref_inSent_var_layer_dict, spr_var_layer_dict, srl_var_layer_dict, ner_var_layer_dict,nonterm_var_layer_dict, ylabel="Variance of Layer")
    # plots_2_tasks(coref_best_layer_dict, coref_inSent_best_layer_dict, spr_best_layer_dict, srl_best_layer_dict,ner_best_layer_dict,nonterm_best_layer_dict, ylabel="Best Layer")
    #
    # plots_1_task(coref_best_layer_dict, coref_inSent_best_layer_dict, spr_best_layer_dict, srl_best_layer_dict, ner_best_layer_dict,nonterm_best_layer_dict, ylabel="Best Layer")
    # plots_1_task(coref_num_examples, coref_inSent_num_examples, spr_num_examples, srl_num_examples, ner_num_examples, nonterm_num_examples, ylabel="Number of examples")
    # plots_1_task(coref_first_negative_delta_dict, coref_inSent_first_negative_delta_dict, spr_first_negative_delta_dict, srl_first_negative_delta_dict, ner_first_negative_delta_dict,nonterm_first_negative_delta_dict, "layer where starts negative delta")



if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)