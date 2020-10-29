import sys, os, re, json
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jiant import *
from utils import *

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


def plots_2_tasks(coref, spr, srl, ner, nonterminals, ylabel=""):
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
    # SPR vs. co-reference
    plot(coref, spr, 'co-reference', 'SPR',
         MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_SPR_THRESHOLD_DISTANCE,
         'Threshold distance', ylabel, title, True, True,224)
    # non-terminals vs. in-sentence co-reference
   #  plot(coref_inSent, nonterminals, 'co-reference', 'non-terminals',
   #       MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_NONTERMINAL_THRESHOLD_DISTANCE,
   #       'Threshold distance', ylabel, title, True, True,221)
   #  # NER vs. in-sentence co-reference
   #  plot(coref_inSent, ner, 'co-reference', 'NER',
   #       MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_NER_THRESHOLD_DISTANCE,
   #       'Threshold distance', ylabel, title, True, True,222)
   #  # srl vs. in-sentence co-reference
   #  plot(coref_inSent, srl, 'co-reference', 'SRL',
   #      MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_SRL_THRESHOLD_DISTANCE,
   #      'Threshold distance', ylabel, title, True, True,223)
   #  # SPR vs. in-sentence co-reference
   #  plot(coref_inSent, spr, 'co-reference', 'SPR',
   #       MAX_COREF_NEW_THRESHOLD_DISTANCE, MAX_SPR_THRESHOLD_DISTANCE,
   #       'Threshold distance', ylabel, title, True, True,224)
   # # co-reference vs. in-sentence co-reference
   #  plot(coref, coref_inSent, 'co-reference', 'in-sentence co-reference',
   #       MAX_COREF_OLD_THRESHOLD_DISTANCE, MAX_COREF_NEW_THRESHOLD_DISTANCE,
   #       'Threshold distance', ylabel, title, True, True,111)


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

    def max_exp(df, task):
        return max(df.loc[df['task'] == task][y_label])

    def min_exp(df, task):
        return min(df.loc[df['task'] == task][y_label])

    def get_task_order(new_df):
        return list(pd.DataFrame({t:max_exp(new_df, t) for t in new_df['task']}, index=[0]).transpose().sort_values(by=0).index)
        # ordered_tasks_df = new_df.loc[new_df[x_label] == '0-2'] if x_label=='spans' else new_df.loc[new_df[x_label] == 1]
        # ordered_tasks_df = ordered_tasks_df.sort_values(by=y_label)
        # return list(ordered_tasks_df['task'])

    new_df = get_new_df(span_exp_layer)
    task_order_list = get_task_order(new_df)
    task_order_list.reverse()
    new_df = new_df.set_index('task').loc[task_order_list].reset_index()
    custom_palette = sns.color_palette("colorblind", 8)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': 25, 'legend.fontsize': 19.5,
          'axes.titlesize': 25, 'xtick.labelsize': 25, 'ytick.labelsize': 25, "lines.linewidth": 3, "lines.markersize": 8}
    sns.set(rc=rc)
    lnp = sns.lineplot(x=x_label, y=y_label, data=new_df, hue="task",
                 style="task", palette=sns.set_palette(custom_palette), dashes=False,
                 markers=["o", "<", ">", "*", "d", "X" , "s"], legend="brief", )
    #plt.setp(lnp.get_legend().get_texts(), fontsize='25')  # for legend text
    axes = lnp.axes
    axes.set_xlim(-0.1, xlim + 0.1)

    if barGraphToo:
        def get_min_max_df(new_df):
            min_max = pd.DataFrame([{'task': t, 'min': min_exp(new_df, t), 'max': max_exp(new_df, t)} for t in new_df['task']]).drop_duplicates()
            min_max_df = min_max[['task', 'min']].rename(columns = {'min': 'expected layer'}, inplace = False)
            min_max_df = min_max_df.append(min_max[['task', 'max']].rename(columns = {'max': 'expected layer'}, inplace = False))
            return min_max_df

        task_order_list.reverse()
        task_order_dict = {task: task_order_list.index(task) for task in task_order_list}
        min_max_df = get_min_max_df(new_df)


        fig, ax = plt.subplots()
        tmp_df = list(min_max_df.loc[min_max_df['task']=='co-reference']['expected layer'])
        for i,task in enumerate(min_max_df['task'].unique()):
            if 'non-terminals' in task:
                updated_task = 'non-\nterminals'
            else:
                updated_task = task
            tmp_df = list(min_max_df.loc[min_max_df['task']==task]['expected layer'])
            start = tmp_df[0]
            diff = tmp_df[1] - tmp_df[0]
            ax.broken_barh([(start, diff)], (task_order_dict[task]*1.2, 1), facecolors=custom_palette[i])
            ax.text(x=start + diff/2,
                    y=task_order_dict[task]*1.2 + 0.5,
                    s=updated_task,
                    ha='center',
                    va='center',
                    color='black',
                    size='21',
                    )
        plt.xticks(np.arange(0, 6, step=0.3))
        ax.set_xlim(1.6, 5.3)
        ax.set_xlabel('excpected layer')
        ax.get_yaxis().set_visible(False)
        ax.grid(True)
        #plt.title("Expected Layer Ranges")
        plt.show()

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
    # total_df = total_df.loc[(total_df['tasks'] == 'SPR to relations') | (total_df['tasks'] == 'SRL to dependencies') | (
    #             total_df['tasks'] == 'co-reference to relations')]
    # total_df = total_df.replace('SPR to relations', 'E(SPR)-E(relations)\n(for NDE - \nSPR span distribution)')
    # total_df = total_df.replace('SRL to dependencies', 'E(dependencies)-E(SRL)\n(for NDE - \nSRL span distribution)')
    # total_df = total_df.replace('co-reference to relations',
    #                             'E(relations)-E(co-reference)\n(for NDE - \nco-reference span distribution)')
    # total_df['result'] = pd.DataFrame.abs(pd.to_numeric(total_df['result']))
    # total_df.index = range(total_df.shape[0])
    # total_df.at[0, 'result'] = str(-1 * float(total_df.at[0, 'result']))
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': 25, 'legend.fontsize': 20,
          'axes.titlesize': 25, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
    sns.set(rc=rc)
    lnp = sns.barplot(x='tasks', y='result', data=total_df, hue="value", palette="colorblind")

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
    rc = {'font.size': 25, 'axes.labelsize': 25, 'legend.fontsize': 14,
          'axes.titlesize': 25, 'xtick.labelsize': 25, 'ytick.labelsize': 25}
    sns.set(rc=rc)
    sns.barplot(x="task_order", y='diff between exp layers', data=new_df, hue="task",
                palette="colorblind", )

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
    rc = {'font.size': 25, 'axes.labelsize': 25, 'legend.fontsize': 16,
          'axes.titlesize': 25, 'xtick.labelsize': 25, 'ytick.labelsize': 25}
    sns.set(rc=rc)
    sns.barplot(x="spans", y='Expected Layer', data=new_df, hue="task",
                palette="colorblind", )
    xlabels = tuple(pd.Series.unique(new_df['spans']))
    plt.xticks(np.arange(len(xlabels)) - 0.2, xlabels, rotation=0, fontsize="18.5", va="center")

