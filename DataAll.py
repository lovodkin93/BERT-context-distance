import sys, os, re, json
from importlib import reload
import itertools
import collections
import numpy as np
import pandas as pd

import jiant.probing.analysis as analysis
reload(analysis)

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

class Data:

    def __init__(self, path, max_thr, name):
        self.name = name
        self.data_path = path
        self.max_thr = max_thr
        self.data_df = self.get_data(path, max_thr)

    def agg_stratifier_group(self, df, stratifier, key_predicate, group_name):
        agg_map = {k: "sum" for k in df.columns if k.endswith("_count")}
        # Use this for short-circuit evaluation, so we don't call key_predicate on invalid keys
        mask = [(s == stratifier and key_predicate(key))
                for s, key in zip(df['stratifier'], df['stratum_key'])]
        sdf = df[mask].groupby(by=ID_COLS).agg(agg_map).reset_index()
        sdf['label'] = group_name
        return sdf

    def load_scores_file(self, filename, tag=None, seed=None, max_threshold_distance=20):
        df = pd.read_csv(filename, sep="\t", header=0)
        df.drop(['Unnamed: 0'], axis='columns', inplace=True)
        # df['task_raw'] = df['task'].copy()
        df['task'] = df['task'].map(analysis.clean_task_name)
        if not "stratifier" in df.columns:
            df["stratifier"] = None
        if not "stratum_key" in df.columns:
            df["stratum_key"] = 0
        # Custom aggregations - Span distances - for every THRESHOLD_DISTANCE between 1 and MAX_SPR_THRESHOLD_DISTANCE split into bigger than THRESHOLD_DISTANCE and smaller than it
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1):
            _eg = []
            _eg.append(self.agg_stratifier_group(df, 'span_distance', lambda x: int(x) <= THRESHOLD_DISTANCE,
                                            f'at_most_{THRESHOLD_DISTANCE}_span1_span2_dist'))
            _eg.append(self.agg_stratifier_group(df, 'span_distance', lambda x: int(x) >= THRESHOLD_DISTANCE,
                                            f'at_least_{THRESHOLD_DISTANCE}_span1_span2_dist'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1 - CASUAL_EFFECT_SPAN_SIZE,
                                        CASUAL_EFFECT_SPAN_SIZE):
            _eg = []
            l_bound = THRESHOLD_DISTANCE  # lower bound
            h_bound = THRESHOLD_DISTANCE + CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
            _eg.append(self.agg_stratifier_group(df, 'span_distance', lambda x: l_bound <= int(x) <= h_bound,
                                            f'{l_bound}-{h_bound}_span1_span2_dist'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1):
            _eg = []
            _eg.append(self.agg_stratifier_group(df, 'span1_length', lambda x: int(x) <= THRESHOLD_DISTANCE,
                                            f'at_most_{THRESHOLD_DISTANCE}_span1_len'))
            _eg.append(self.agg_stratifier_group(df, 'span1_length', lambda x: int(x) >= THRESHOLD_DISTANCE,
                                            f'at_least_{THRESHOLD_DISTANCE}_span1_len'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1 - NER_CASUAL_EFFECT_SPAN_SIZE,
                                        NER_CASUAL_EFFECT_SPAN_SIZE):
            _eg = []
            l_bound = THRESHOLD_DISTANCE  # lower bound
            h_bound = THRESHOLD_DISTANCE + NER_CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
            _eg.append(self.agg_stratifier_group(df, 'span1_length', lambda x: l_bound <= int(x) <= h_bound,
                                            f'{l_bound}-{h_bound}_span1_len'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1):
            _eg = []
            _eg.append(self.agg_stratifier_group(df, 'span1_span2_length', lambda x: int(x) <= THRESHOLD_DISTANCE,
                                            f'at_most_{THRESHOLD_DISTANCE}_span1_span2_len'))
            _eg.append(self.agg_stratifier_group(df, 'span1_span2_length', lambda x: int(x) >= THRESHOLD_DISTANCE,
                                            f'at_least_{THRESHOLD_DISTANCE}_span1_span2_len'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)
        for THRESHOLD_DISTANCE in range(0, max_threshold_distance + 1 - NER_CASUAL_EFFECT_SPAN_SIZE,
                                        NER_CASUAL_EFFECT_SPAN_SIZE):
            _eg = []
            l_bound = THRESHOLD_DISTANCE  # lower bound
            h_bound = THRESHOLD_DISTANCE + NER_CASUAL_EFFECT_SPAN_SIZE - 1  # higher bound is minus 1 of the next loewer bound
            _eg.append(self.agg_stratifier_group(df, 'span1_span2_length', lambda x: l_bound <= int(x) <= h_bound,
                                            f'{l_bound}-{h_bound}_span1_span2_len'))
            df = pd.concat([df] + _eg, ignore_index=True, sort=False)

        df.insert(0, "exp_name", df['run'].map(lambda p: os.path.basename(os.path.dirname(p.strip("/")))))
        df.insert(1, "exp_type", df['exp_name'].map(analysis.get_exp_type))
        df.insert(1, "layer_num", df['exp_name'].map(analysis.get_layer_num))
        df.insert(11, "total_count", df['fn_count'] + df['fp_count'] + df['tn_count'] + df['tp_count'])
        if tag is not None:
            df.insert(0, "tag", tag)
        df.insert(1, "seed", seed)
        return df

    def _format_display_col(self, exp_type, layer_num, tag):
        ret = exp_type
        if layer_num:
            ret += f"-{layer_num}"
        if tag:
            ret += f" ({tag})"
        return ret

    def get_data(self, file_name, max_threshold_distance):
        score_files = [("mix",
                        "/cs/labs/oabend/lovodkin93/jiant_rep/jiant-ep_frozen_20190723/probing/" + file_name)]  # changes from the original Edgeprobe_Aggregate_Analysis.py
        dfs = []
        for tag, score_file in score_files:
            df = self.load_scores_file(score_file, tag=tag, max_threshold_distance=max_threshold_distance)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True, sort=False)

        df['display_col'] = list(map(self._format_display_col, df.exp_type, df.layer_num, df.tag))
        analysis.score_from_confusion_matrix(df)

        def _get_final_score(row):
            return row['f1_score'], row['f1_errn95']

        df['score'], df['score_errn95'] = zip(*(_get_final_score(row) for i, row in df.iterrows()))
        return df