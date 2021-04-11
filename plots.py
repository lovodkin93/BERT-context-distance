import seaborn as sns
from utils import *

def plot_span_expected_layer(span_exp_layer, x_label, y_label, xlim_min, xlim_max ,ylim_min, ylim_max, axes_title_size, xticks_size, yticks_size, legend_size , barGraphToo=True, isSpan=True, isPercent=False):

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
    if isPercent:
        new_df['span probability'] = new_df['span probability'] * 100
    custom_palette = sns.color_palette("colorblind", 8)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 25, 'axes.labelsize': axes_title_size, 'legend.fontsize': legend_size,
          'axes.titlesize': 33, 'xtick.labelsize': xticks_size, 'ytick.labelsize': yticks_size, "lines.linewidth": 2,
          "lines.markersize": 3}
    # params = {'text.latex.preamble' : [r'\usepackage{amsmath, amssymb}'], }
    # plt.rcParams.update(params)

    sns.set(rc=rc)
    lnp = sns.lineplot(x=x_label, y=y_label, data=new_df, hue="task",
                       style="task", palette=sns.set_palette(custom_palette), dashes=False,
                       markers=["o", "<", ">", "*", "d", "X", "s"], legend="brief", )
    axes = lnp.axes
    plt.gca().legend().set_title('')
    plt.ylabel('Span Distribution (%)')
    axes.set_xlim(xlim_min - 0.1, xlim_max + 0.1)
    axes.set_ylim(ylim_min, ylim_max)
    plt.savefig('./figures/' + y_label + '_' + x_label + '.png', bbox_inches='tight', dpi=300)

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
        tmp_df = list(min_max_df.loc[min_max_df['task'] == 'CO-REF.']['expected layer'])
        for i, task in enumerate(min_max_df['task'].unique()):
            if 'NON-TERM.' in task:
                updated_task = 'NON-\nTERM.'
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
                    size='3.5',
                    )
        plt.xticks(np.arange(0, 6, step=0.5), fontsize=5)
        ax.set_xlim(0.22, 4.65)
        ax.set_ylim(-0.15, 8.38)
        ax.set_xlabel('$\mathbb{E}_{layer}$', fontsize=7)
        ax.get_yaxis().set_visible(False)
        ax.grid(True)
        for t in ax.get_xticklabels():
            t.set_transform(t.get_transform() + mtrans.Affine2D().translate(0, 30))
        plt.savefig('./figures/bars.png', bbox_inches='tight', dpi=300)
        # plt.title("Expected Layer Ranges")

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


def plot_TCE_NDE_NIE(TCE, NDE, NIE, exp_layer_diff, specific_tasks=None, noTCE=False, noNDE=False, noNIE=False, noExpLayerDiff=False, fig_name = 'NDE_vs_Umediated', allData=False):

    def get_relevant_df(dict, name):
        df = pd.DataFrame(dict)
        df['result'] = df[df.columns[0:]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
        )
        df['tasks'] = df.index
        df['values'] = name
        df = df[['values', 'tasks', 'result']]
        return df

    def clean_df(total_df):
        cleaned_df = total_df.loc[total_df['tasks'].isin(specific_tasks)]
        cleaned_df = cleaned_df.loc[cleaned_df['values'] != 'TCE'] if noTCE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['values'] != 'NDE'] if noNDE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['values'] != 'NIE'] if noNIE else cleaned_df
        cleaned_df = cleaned_df.loc[cleaned_df['values'] != 'unmediated'] if noExpLayerDiff else cleaned_df
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
        total_df.index = total_df.index + " " + total_df['values']
        for task_pair in pairs_to_update:
            nde_index = task_pair + " NDE"
            exp_index = task_pair + " unmediated" if noTCE else task_pair + " TCE"
            total_df.at[nde_index, 'result'] = str(-1 * float(total_df.at[nde_index, 'result']))
            total_df.at[exp_index, 'result'] = str(-1 * float(total_df.at[exp_index, 'result']))
        return total_df

    def get_ordered(total_df, task_sub_list):
        change_task_sub_list_dict = {k: abs(float(total_df.loc[total_df['tasks'] == k].loc[total_df['values'] == 'unmediated']['Difference in Expected Layers']) -
                                  float(total_df.loc[total_df['tasks'] == k].loc[total_df['values'] == 'NDE']['Difference in Expected Layers']))
                           for k in task_sub_list}
        ordered_task_sub_list = nlargest(len(task_sub_list), task_sub_list, key=change_task_sub_list_dict.get)
        return ordered_task_sub_list

    def order_tasks(total_df):
        task_pair_list = list(set([elem for elem in total_df['tasks']]))
        changed = []
        dec = []
        inc = []
        for task_pair in task_pair_list:
            tmp = total_df.loc[total_df['tasks'] == task_pair]
            if float(tmp['Difference in Expected Layers'][0]) * float(tmp['Difference in Expected Layers'][1]) < 0:
                changed.append(task_pair)
            elif float(tmp['Difference in Expected Layers'][0]) < float(tmp['Difference in Expected Layers'][1]):
                inc.append(task_pair)
            else:
                dec.append(task_pair)
        ordered_changed_list = get_ordered(total_df, changed)
        ordered_dec_list = get_ordered(total_df, dec)
        ordered_inc_list = get_ordered(total_df, inc)
        ordered_tasks_list = ordered_changed_list + ordered_dec_list + ordered_inc_list
        return ordered_tasks_list



    total_df = get_total_df()
    relevant_tasks = list(specific_tasks.keys()) if allData else get_relevant_task()
    total_df = total_df.loc[total_df['tasks'].isin(relevant_tasks)] #filter just relevant tasks
    total_df = update_names(relevant_tasks, total_df)
    total_df = update_neg_values(total_df, relevant_tasks)
    total_df['result'] = pd.to_numeric(total_df['result'])
    total_df = total_df.sort_values(by=['values'], ascending=False)
    total_df = total_df.rename({"result": "Difference in Expected Layers"}, axis='columns')
    ordered_tasks_list = order_tasks(total_df)

    def task_sorter(column):
        correspondence = {task: order for order, task in enumerate(ordered_tasks_list)}
        return column.map(correspondence)

    total_df = total_df.sort_values(by='tasks', key=task_sorter)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 10, 'axes.labelsize': 15, 'legend.fontsize': 9,
          'axes.titlesize': 33, 'xtick.labelsize': 4, 'ytick.labelsize': 12}
    sns.set(rc=rc)
    lnp = sns.barplot(x='tasks', y='Difference in Expected Layers', data=total_df, hue="values", palette="colorblind")
    plt.ylabel('Difference in $\mathbb{E}_{layer}$')
    plt.gca().legend().set_title('')
    plt.xlabel('Tasks')
    plt.ylabel('$\mathbb{E}_{layer}$ Difference')

    plt.xticks(rotation=45)
    plt.savefig('./figures/ ' + fig_name + '.png', bbox_inches='tight', dpi=300)

def plot_diffs_max_min(diffs_max_min):
    import math
    def relevant_sub_df(diffs_max_min, task):
        diffs_max_min_df = pd.DataFrame(diffs_max_min)
        diffs_max_min_df['Task Order'] = diffs_max_min_df.index
        sub_df = pd.DataFrame(diffs_max_min_df[['Task Order', task]])
        sub_df['task'] = task
        sub_df = sub_df.rename(columns={task: 'Difference between $\mathbb{E}_{layer}$'})
        sub_df = sub_df.loc[sub_df['Difference between $\mathbb{E}_{layer}$'] == sub_df['Difference between $\mathbb{E}_{layer}$']] #to get rid of None
        seperate_tasks = task.split('-')
        sub_df['left task'] = seperate_tasks[0]
        sub_df['right task'] = seperate_tasks[1]
        return sub_df

    def get_new_df(diffs_max_min):
        new_df = pd.DataFrame(columns=['Difference between $\mathbb{E}_{layer}$', 'task', 'left task', 'right task'])
        diffs_max_min_df = pd.DataFrame(diffs_max_min)
        for task in diffs_max_min_df.columns:
            new_df = new_df.append(relevant_sub_df(diffs_max_min_df, task))
        return new_df

    def order_tasks(total_df):
        task_pair_list = list(set([elem for elem in total_df['task']]))
        changed = []
        unchaged = []
        for task_pair in task_pair_list:
            tmp = total_df.loc[total_df['task'] == task_pair]
            if float(tmp['Difference between $\mathbb{E}_{layer}$'][0]) * float(tmp['Difference between $\mathbb{E}_{layer}$'][1]) < 0:
                changed.append(task_pair)
            else:
                unchaged.append(task_pair)
        ordered_tasks_list = changed + unchaged
        return ordered_tasks_list

    new_df = get_new_df(diffs_max_min)
    new_df = get_new_df(diffs_max_min)
    ordered_tasks_list = order_tasks(new_df)

    def task_sorter(column):
        correspondence = {task: order for order, task in enumerate(ordered_tasks_list)}
        return column.map(correspondence)

    new_df = new_df.sort_values(by='task', key=task_sorter)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 10, 'axes.labelsize': 13, 'legend.fontsize': 5,
          'axes.titlesize': 33, 'xtick.labelsize': 4, 'ytick.labelsize': 13}
    sns.set(rc=rc)
    new_df_changed = new_df.rename(columns={'task': 'Task Pair'}, inplace=False)
    lnp = sns.barplot(x="Task Pair", y='Difference between $\mathbb{E}_{layer}$', data=new_df_changed, hue="Task Order",
                      palette="colorblind", )
    axes = lnp.axes
    # axes.set_xlim(-0.5, 2.3)
    axes.set_ylim(-4.4, 2.7)
    plt.gca().legend().set_title('')
    plt.xticks(rotation=45)
    plt.xlabel('Task-Pairs')
    plt.ylabel('$\mathbb{E}_{layer}$ Difference')
    plt.savefig('./figures/max_min.png', bbox_inches='tight', dpi=300)

def plot_sympson_paradox(span_dict, simple_task, complex_task, specific_span_simple, specific_span_complex, ylim_max):
    def relevant_sub_df(df, task):
        span_df = pd.DataFrame(span_dict)
        span_df['Context Length Ranges'] = span_df.index
        sub_df = pd.DataFrame(span_df[['Context Length Ranges', task]])
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
        return task + ' (Context Length$\in$' + span + ')'

    new_df = get_new_df(span_dict)
    new_df = new_df.loc[(new_df['task'] == simple_task) | (new_df['task'] == complex_task)]
    new_df = new_df.append(
        {'task': get_special_name(complex_task, specific_span_complex),
         'Context Length Ranges': 'Context Length\n Distribution',
         'Expected Layer': span_dict[complex_task][specific_span_complex]}, ignore_index=True)
    new_df = new_df.append(
        {'task': get_special_name(simple_task, specific_span_simple),
         'Context Length Ranges': 'Context Length\n Distribution',
         'Expected Layer': span_dict[simple_task][specific_span_simple]}, ignore_index=True)
    plt.figure(figsize=(16, 9))
    sns.set(style='darkgrid', )
    rc = {'font.size': 10, 'axes.labelsize': 15, 'legend.fontsize': 7.5,
          'axes.titlesize': 33, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
    sns.set(rc=rc)
    ax = sns.barplot(x='Context Length Ranges', y='Expected Layer', data=new_df, hue="task",
                     palette="colorblind", )
    plt.gca().legend().set_title('')
    trans = mtrans.Affine2D().translate(50, 0)
    for t in ax.get_xticklabels():
        if 'Context Length' in t._text:
            # t.set_rotation(20)
            t.set_transform(t.get_transform() + mtrans.Affine2D().translate(60, 0))
        else:
            t.set_transform(t.get_transform() - trans)
    plt.ylabel('$\mathbb{E}_{layer}$')
    ax.set_ylim(0, ylim_max)
    plt.savefig('./figures/simpson_' + simple_task + '_' + complex_task + '.png', bbox_inches='tight', dpi=300)

