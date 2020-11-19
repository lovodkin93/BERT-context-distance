import seaborn as sns
from utils import *

def plot_span_expected_layer(span_exp_layer, x_label, y_label, xlim_min, xlim_max ,ylim_min, ylim_max , barGraphToo=True, isSpan=True):

    def relevant_sub_df(span_exp_layer, task):
        span_exp_layer_df = pd.DataFrame(span_exp_layer)
        span_exp_layer_df[x_label] = span_exp_layer_df.index
        sub_df = pd.DataFrame(span_exp_layer_df[[x_label, task]])
        sub_df['task'] = task
        sub_df = sub_df.rename(columns={task: y_label})
        return sub_df

    def reorder_df(df): #needed to move the "leftovers" to the end
        m = df.index.str.find('-') != -1 # check in what line the index is of the structure "NUM - NUM"
        df = df[m].append(df[~m]).reset_index(drop=True)
        df.index = list(span_exp_layer['SPR'].keys()) #choosing the "SPR" at random - just need the original spans order
        return df

    def get_new_df(span_exp_layer):
        new_df = pd.DataFrame(columns=[x_label, y_label, 'task'])
        span_exp_layer_df = pd.DataFrame(span_exp_layer)
        span_exp_layer_df = reorder_df(span_exp_layer_df) if isSpan else span_exp_layer_df
        for task in span_exp_layer_df.columns:
            new_df = new_df.append(relevant_sub_df(span_exp_layer_df, task))
        return new_df

    def max_exp(df, task):
        return max(df.loc[df['task'] == task][y_label])

    def min_exp(df, task):
        return min(df.loc[df['task'] == task][y_label])

    def get_task_order(new_df):
        return list(pd.DataFrame({t:max_exp(new_df, t) for t in new_df['task']}, index=[0]).transpose().sort_values(by=0).index)

    new_df = get_new_df(span_exp_layer)
    task_order_list = get_task_order(new_df)
    task_order_list.reverse()
    new_df = new_df.set_index('task').loc[task_order_list].reset_index()
    custom_palette = sns.color_palette("colorblind", 8)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': 33, 'legend.fontsize': 21.5,
          'axes.titlesize': 33, 'xtick.labelsize': 30, 'ytick.labelsize': 30, "lines.linewidth": 3,
          "lines.markersize": 8}
    sns.set(rc=rc)
    lnp = sns.lineplot(x=x_label, y=y_label, data=new_df, hue="task",
                       style="task", palette=sns.set_palette(custom_palette), dashes=False,
                       markers=["o", "<", ">", "*", "d", "X", "s"], legend="brief", )
    axes = lnp.axes
    plt.gca().legend().set_title('')
    axes.set_xlim(xlim_min - 0.1, xlim_max + 0.1)
    axes.set_ylim(ylim_min, ylim_max)

    if barGraphToo:
        def get_min_max_df(new_df):
            min_max = pd.DataFrame([{'task': t, 'min': min_exp(new_df, t), 'max': max_exp(new_df, t)} for t in
                                    new_df['task']]).drop_duplicates()
            min_max_df = min_max[['task', 'min']].rename(columns={'min': 'expected layer'}, inplace=False)
            min_max_df = min_max_df.append(
                min_max[['task', 'max']].rename(columns={'max': 'expected layer'}, inplace=False))
            return min_max_df

        task_order_list.reverse()
        task_order_dict = {task: task_order_list.index(task) for task in task_order_list}
        min_max_df = get_min_max_df(new_df)
        fig, ax = plt.subplots()
        tmp_df = list(min_max_df.loc[min_max_df['task'] == 'co-reference']['expected layer'])
        for i, task in enumerate(min_max_df['task'].unique()):
            if 'non-terminals' in task:
                updated_task = 'non-\nterminals'
            else:
                updated_task = task
            tmp_df = list(min_max_df.loc[min_max_df['task'] == task]['expected layer'])
            start = tmp_df[0]
            diff = tmp_df[1] - tmp_df[0]
            ax.broken_barh([(start, diff)], (task_order_dict[task] * 1.2, 1), facecolors=custom_palette[i])
            ax.text(x=start + diff / 2,
                    y=task_order_dict[task] * 1.2 + 0.5,
                    s=updated_task,
                    ha='center',
                    va='center',
                    color='black',
                    size='21',
                    )
        plt.xticks(np.arange(0, 6, step=0.3))
        ax.set_xlim(0.3, 4.58)
        ax.set_ylim(-0.05, 8.25)
        ax.set_xlabel('expected layer')
        ax.get_yaxis().set_visible(False)
        ax.grid(True)
        #plt.title("Expected Layer Ranges")

def plot_CDE(cde):

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

    initial_subplt_orient = 100 * (find_divider(len(cde))) + 10 * (len(cde) // find_divider(len(cde)))
    for idx, k in enumerate(cde.keys()):
        subplt_orient = initial_subplt_orient + idx + 1
        plot_cde(cde[k], '-' + k + '-', subplt_orient)
    plt.show()

def plot_all_CDE(CDE_all):
    for task_CDE in CDE_all.values():
        plot_CDE(task_CDE)


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
        cleaned_df = cleaned_df.loc[cleaned_df['value'] != 'unmediated'] if noExpLayerDiff else cleaned_df
        return cleaned_df


    def get_total_df():
        TCE_df = get_relevant_df(TCE, 'TCE')
        NDE_df = get_relevant_df(NDE, 'NDE')
        NIE_df = get_relevant_df(NIE, 'NIE')
        exp_layer_diff_df = get_relevant_df(exp_layer_diff, 'unmediated')
        total_df = pd.concat([TCE_df, NDE_df, NIE_df, exp_layer_diff_df])
        total_df = clean_df(total_df)
        return total_df

    def get_relevant_task():
        incr_tasks = {k: specific_tasks[k]['change(%)'] for k in specific_tasks.keys() if specific_tasks[k]['incr/decr'] == 'increase' and specific_tasks[k]['change difficulty balance'] == False}
        dec_tasks = {k: specific_tasks[k]['change(%)'] for k in specific_tasks.keys() if specific_tasks[k]['incr/decr'] == 'decrease' and specific_tasks[k]['change difficulty balance'] == False}
        change_diff_bal_tasks = {k: specific_tasks[k]['change(abs val)'] for k in specific_tasks.keys() if specific_tasks[k]['change difficulty balance'] == True}
        max_incr_task = max(incr_tasks, key=incr_tasks.get)
        max_dec_task = max(dec_tasks, key=dec_tasks.get)

        incr_tasks_no_max = {k: incr_tasks[k] for k in incr_tasks.keys() if k != max_incr_task}
        dec_tasks_no_max = {k: dec_tasks[k] for k in dec_tasks.keys() if k != max_dec_task}
        second_max_incr_task = max(incr_tasks_no_max, key=incr_tasks_no_max.get) if len(incr_tasks_no_max) else None
        second_max_dec_task = max(dec_tasks_no_max, key=dec_tasks_no_max.get) if len(dec_tasks_no_max) else None

        if len(change_diff_bal_tasks):
            third_task = max(change_diff_bal_tasks, key=change_diff_bal_tasks.get)
        elif second_max_incr_task and second_max_dec_task:
            third_task = second_max_incr_task if (incr_tasks[second_max_incr_task] > dec_tasks[second_max_dec_task]) else second_max_dec_task
        else:
            third_task = second_max_incr_task if second_max_incr_task else second_max_dec_task
        return max_incr_task, max_dec_task, third_task

    def update_names(relevant_tasks, total_df):
        for old_name in relevant_tasks:
            tasks = re.split(" to ", old_name)
            res = total_df.loc[total_df['tasks'] == old_name]['result']
            c = 1
            if (float(res.values[0]) < 0 and float(res.values[1]) < 0):
                new_name = tasks[0] + ' - ' + tasks[1]
            else:
                new_name = tasks[1] + ' - ' + tasks[0]
            total_df = total_df.replace(old_name, new_name)
        return total_df

    def update_neg_values(total_df, relevant_tasks):

        pairs_to_update = list()
        for task_pair in relevant_tasks:
            results = total_df.loc[total_df.index == task_pair]['result']
            if float(results.values[0]) < 0 and float(results.values[1]) < 0:
                pairs_to_update.append(task_pair)
        total_df.index = total_df.index + " " + total_df['value']
        for task_pair in pairs_to_update:
            nde_index = task_pair + " NDE"
            exp_index = task_pair + " unmediated" if noTCE else task_pair + " TCE"
            total_df.at[nde_index, 'result'] = str(-1 * float(total_df.at[nde_index, 'result']))
            total_df.at[exp_index, 'result'] = str(-1 * float(total_df.at[exp_index, 'result']))
        return total_df

    total_df = get_total_df()
    relevant_tasks = get_relevant_task()
    total_df = total_df.loc[(total_df['tasks'] == relevant_tasks[0]) | (total_df['tasks'] == relevant_tasks[1]) | (
                total_df['tasks'] == relevant_tasks[2])]
    total_df = update_names(relevant_tasks, total_df)
    total_df = update_neg_values(total_df, relevant_tasks)
    total_df['result'] = pd.to_numeric(total_df['result'])
    total_df = total_df.sort_values(by=['value'], ascending=False)
    total_df = total_df.rename({"result": "Difference in Expected Layers"}, axis='columns')
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': 33, 'legend.fontsize': 24,
          'axes.titlesize': 33, 'xtick.labelsize': 30, 'ytick.labelsize': 30}
    sns.set(rc=rc)
    lnp = sns.barplot(x='tasks', y='Difference in Expected Layers', data=total_df, hue="value", palette="colorblind")
    plt.gca().legend().set_title('')

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

def plot_sympson_paradox(span_dict, simple_task, complex_task, specific_span_simple, specific_span_complex, ylim_max):
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

    def get_special_name(task, span):
        return task + ' (span: ' + span + ')'

    new_df = get_new_df(span_dict)
    new_df = new_df.loc[(new_df['task'] == simple_task) | (new_df['task'] == complex_task)]
    new_df = new_df.append({'task': get_special_name(complex_task, specific_span_complex), 'spans': '',
                            'Expected Layer': span_dict[complex_task][specific_span_complex]}, ignore_index=True)
    new_df = new_df.append({'task': get_special_name(simple_task, specific_span_simple), 'spans': '',
                            'Expected Layer': span_dict[simple_task][specific_span_simple]}, ignore_index=True)
    # new_df = new_df.append({'task': 'co-reference (span: 0-2)', 'spans': '', 'Expected Layer': span_dict[complex_task]['0-2']}, ignore_index=True)
    # new_df = new_df.append({'task': 'NER (span: 9+)', 'spans': '', 'Expected Layer': span_dict[simple_task]['9+']}, ignore_index=True)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': 33, 'legend.fontsize': 23,
          'axes.titlesize': 33, 'xtick.labelsize': 30, 'ytick.labelsize': 30}
    sns.set(rc=rc)
    ax = sns.barplot(x="spans", y='Expected Layer', data=new_df, hue="task",
                     palette="colorblind", )
    plt.gca().legend().set_title('')
    trans = mtrans.Affine2D().translate(60, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() - trans)
    ax.set_ylim(0, ylim_max)


