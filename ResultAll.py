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
    def __init__(self, data_class, max_thr, at_most_least, context_distance, dataAll_class):
        # dict - for threshold, no dict - the general case - when looking at all instances
        self.data_class = data_class
        df = data_class.data_df.loc[ (data_class.data_df['label'] == '_micro_avg_') & (data_class.data_df['split'] == SPLIT)]
        self.expected_layer, self.first_neg_delta, self.var_layer, self.best_num_layer = calc_expected_layer(df)
        self.expected_layer_dict, self.first_neg_delta_dict, self.var_layer_dict, self.best_layer_dict, self.num_examples_dict = self.calculate_thr_dict_values(data_class.data_df, max_thr, span=context_distance, at_most_least=at_most_least)
        self.span_exp_layer, self.span_prob =  TCE_helper(data_class.data_df, MAX_ALL_THRESHOLD_DISTANCE, allSpans=ALL_SPANS, span=context_distance) #FIXME: update functions calling TCE_helper so no double calling (in utils)
        self.TCE, self.CDE, self.NDE, self.NIE = self.calculate_TCE_CDE_NIE_NDE(data_class, context_distance, data_class.name, max_thr, dataAll_class)

    def calculate_thr_dict_values(self, df, max_threshold_distance, span=TWO_SPANS_SPAN, at_most_least=AT_MOST):
        # span = span type: span1 length (if only span1), distance between span1 and span 2 or the total length of span1
        #           to span2 (or vice versa). can be SPAN1_LEN, SPAN1_SPAN2_LEN or SPAN1_SPAN2_DIST
        # at_most_least = whether the close threshold (<=thr) or the far threshold (>=thr)
        exp_layer_dict, var_layer_dict, first_negative_delta_dict, best_layer_dict, num_examples_dict = dict(), dict(), dict(), dict(), dict()
        for THRESHOLD_DISTANCE in range(2, max_threshold_distance):
            curr_df = df.loc[(df['label'] == f'{at_most_least}_{THRESHOLD_DISTANCE}_{span}') & (df['split'] == SPLIT)]
            num_examples_dict[THRESHOLD_DISTANCE] = curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0]
            if curr_df.loc[curr_df['layer_num'] == '0']['total_count'].values[0] > MIN_EXAMPLES_CNT:
                exp_layer_dict[THRESHOLD_DISTANCE], first_negative_delta_dict[THRESHOLD_DISTANCE], var_layer_dict[
                    THRESHOLD_DISTANCE], best_layer_dict[THRESHOLD_DISTANCE] = calc_expected_layer(curr_df)
        return exp_layer_dict, first_negative_delta_dict, var_layer_dict, best_layer_dict, num_examples_dict

    def calculate_TCE_CDE_NIE_NDE(self, task_data, context_distance, task_name, max_thr, dataAll_class):
        TCE, CDE, NDE, NIE = dict(), dict(), dict(), dict()
        a = vars(dataAll_class)
        b = list()
        for task in vars(dataAll_class).values():
            if task.name != task_data.name:
                a = task_data.name + ' to ' + task.name
                TCE[a], CDE[a], NDE[a], NIE[a] = all_effects(task_data.data_df, task.data_df, max_thr, task.max_thr,
                                                             allSpans=ALL_SPANS, span1=context_distance,
                                                             span2=task.context_distance)
        return TCE, CDE, NDE, NIE

    def get_general_values(self):
        return self.expected_layer, self.first_neg_delta, self.var_layer, self.best_num_layer

    def get_thr_dict_values(self):
        return self.expected_layer_dict, self.first_neg_delta_dict, self.var_layer_dict, self.best_layer_dict, self.num_examples_dict

    def get_TCE_CDE_NIE_NDE(self):
        return self.TCE, self.CDE, self.NDE, self.NIE

    def get_span_values(self):
        return self.span_exp_layer, self.span_prob

    def get_expected_layer_diff(self, all_exp_layers_dict):
        exp_layer_diff_dict = dict()
        for task_name in all_exp_layers_dict:
            if self.data_class.name != task_name:
                name = self.data_class.name + " to " + task_name
                exp_layer_diff_dict[name] = all_exp_layers_dict[task_name] - self.expected_layer
        return exp_layer_diff_dict

class ResultAll:
    def __init__(self, dataAll_class, at_most_least=AT_MOST):
        self.dataAll_class = dataAll_class
        self.nonterminals = Result(dataAll_class.nonterminals, MAX_NONTERMINAL_THRESHOLD_DISTANCE,at_most_least=at_most_least, context_distance=SPAN1_LEN, dataAll_class=dataAll_class)
        self.dependencies = Result(dataAll_class.dependencies, MAX_DEP_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=TWO_SPANS_SPAN, dataAll_class=dataAll_class)
        self.ner = Result(dataAll_class.ner, MAX_NER_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=SPAN1_LEN, dataAll_class=dataAll_class)
        self.srl = Result(dataAll_class.srl, MAX_SRL_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=TWO_SPANS_SPAN, dataAll_class=dataAll_class)
        self.coreference = Result(dataAll_class.coreference, MAX_COREF_OLD_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=TWO_SPANS_SPAN, dataAll_class=dataAll_class)
        self.relations = Result(dataAll_class.relations, MAX_REL_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=TWO_SPANS_SPAN, dataAll_class=dataAll_class)
        self.spr = Result(dataAll_class.spr, MAX_SPR_THRESHOLD_DISTANCE, at_most_least=at_most_least, context_distance=TWO_SPANS_SPAN, dataAll_class=dataAll_class)

    def get_all_expected_layers(self):
        exp_layer_all = dict()
        # for task in vars(self.dataAll_class).values():
        #     df = task.data_df
        #     curr_df = df.loc[(df['label'] == '_micro_avg_') & (df['split'] == SPLIT)]
        #     exp_layer_all[task.name], _, _, _ = calc_expected_layer(curr_df)
        for task in vars(self).values():
            if hasattr(task, 'expected_layer'):
                exp_layer_all[task.data_class.name] = task.expected_layer
        return exp_layer_all

    def get_all_TCE_CDE_NIE_NDE(self):
        TCE_all, CDE_all, NDE_all, NIE_all = dict(), dict(), dict(), dict()
        for task in vars(self).values():
            if hasattr(task, 'NDE'):
                name = task.data_class.name
                TCE_all[name], CDE_all[name], NDE_all[name], NIE_all[name] = task.TCE, task.CDE, task.NDE, task.NIE
        return TCE_all, CDE_all, NDE_all, NIE_all

    def get_all_expected_layer_diff(self):
        exp_layer_diff_dict = dict()
        for task in vars(self).values():
            if hasattr(task, 'data_class'):
                exp_layer_diff_dict[task.data_class.name] = task.get_expected_layer_diff(self.get_all_expected_layers())
        return exp_layer_diff_dict

    def get_all_span_exp_layer_prob(self):
        span_exp_layer, span_prob = dict(), dict()
        for task in vars(self.dataAll_class).values():
            span_exp_layer[task.name], span_prob[task.name] = TCE_helper(task.data_df, MAX_ALL_THRESHOLD_DISTANCE, allSpans=ALL_SPANS, span=task.context_distance)
        return span_exp_layer, span_prob

    def get_all_thr_dict_values(self):
        exp_layer_dict, first_neg_delta_dict, var_layer_dict, best_layer_dict, num_examples_dict = dict(), dict(), dict(), dict(), dict()
        for task in vars(self).values():
            if hasattr(task, 'data_class'):
                name = task.data_class.name
                exp_layer_dict[name], first_neg_delta_dict[name], var_layer_dict[name], \
                                            best_layer_dict[name], num_examples_dict[name] = task.get_thr_dict_values()
        return exp_layer_dict, first_neg_delta_dict, var_layer_dict, best_layer_dict, num_examples_dict

    def get_biggest_exp_layer_diff_NDE_diff(self, isTCE=False):
        # isTCE is for the case when I eant to compare the TCE to the NDE and not the actual expected layer difference
        def calc_change(exp_layer_diff, NDE):
            increase_decrease = 'decrease' if abs(exp_layer_diff) > abs(NDE) else 'increase'
            change_difficulty_balance = True if ((exp_layer_diff > 0 and NDE < 0) or (exp_layer_diff < 0 and NDE > 0)) else False
            return {'incr/decr': increase_decrease, 'change(%)': abs((exp_layer_diff - NDE) / exp_layer_diff) * 100,
                    'change(abs val)': abs(exp_layer_diff - NDE), 'change difficulty balance': change_difficulty_balance}

        TCE_all, _, NDE_all, _ = self.get_all_TCE_CDE_NIE_NDE()
        exp_layer_diff = TCE_all if isTCE else self.get_all_expected_layer_diff()
        change_dict = {k: (calc_change(exp_layer_diff[ref_task][k], NDE_all[ref_task][k])) for ref_task in NDE_all.keys() for k
                       in NDE_all[ref_task].keys()}
        change_perc_dict = {k: change_dict[k]['change(%)'] for k in change_dict.keys()}
        largest_change = nlargest(10, change_dict, key=change_perc_dict.get)
        return {elem: change_dict[elem] for elem in largest_change}

    def impose_max_min(self):
        # returns diffs, a dict whose keys are of structure "k to j" where k and j are grammatical tasks, and the values are lists where each time we subtract k's expected layer from j's, when the first elem of the list j's exp layer is the max of all the spans and k's is the min and the second elem vice versa
        span_exp_layer, _ = self.get_all_span_exp_layer_prob()
        diffs = dict()
        for k in span_exp_layer.keys():
            for j in span_exp_layer.keys():
                if k + ' minus ' + j not in diffs.keys() and j != k:
                    k_vals = span_exp_layer[k].values()
                    j_vals = span_exp_layer[j].values()
                    diffs[j + ' minus ' + k] = {'max minus min': max(j_vals) - min(k_vals),
                                                'min minus max': min(j_vals) - max(k_vals)}
        return diffs


def main(args):
    a = DataAll()
    b = ResultAll(a)
    c = b.get_all_thr_dict_values()
    print('end')



if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)