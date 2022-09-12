
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pk
import scipy
from .utils import find_all_files_in_dir, oj
from bokeh.io import output_notebook, export_png
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
import itertools
import re
import csv 

# params = {'legend.fontsize': 16,
#           'legend.handlelength': 3}

# plt.rcParams.update(params)


def plot(
    xs: list, 
    ys: list=None, 
    xlabels: list=None, 
    ylabels: list=None, 
    titles: list=None,
    fig_title: str=None,
    fig_path: str=None,
    marker: str='x',
    linestyle: str=None,
    plot_type: str='line',
    **kwargs
):
    
    if ys is None: ys = [None] * len(xs)  # for histogram case
    assert len(xs) == len(ys)

    n_plots = len(xs)
    n_side = int(np.ceil(np.sqrt(n_plots)))
    fig, axs = plt.subplots(n_side, n_side)
    if n_side == 1: axs = [[axs],]
    axs = [ax for sub in axs for ax in sub]

    format_elements = [xlabels, ylabels, titles]
    format_elements = [[None] * n_plots if f is None else f for f in format_elements]
    for f in format_elements:
        assert len(f) == n_plots

    for ax, x, y, xl, yl, ti in zip(axs, xs, ys, *format_elements):
        
        if plot_type is 'line':
            ax.plot(
                x, y, 
                color = 'blue',
                linestyle = linestyle,  # solid, dotted
                marker = marker,
                markerfacecolor = 'purple', 
                markersize = 5
            )

        if plot_type is 'hist':
            ax.hist(
                x,
                **kwargs
            )

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ti)

    fig.suptitle(fig_title)
    fig.tight_layout()
    if fig_path is not None: fig.savefig(fig_path)

    return fig



def create_error_str_from_numbers(value, val_error):
    precision = 10**np.floor(np.log10(val_error))
    str_precision = '{:.16f}'.format(precision)
    error = int(val_error / precision)
    split_precision = str_precision.split('.')

    if precision > 1:
        raise NotImplementedError
    else:
        # lens to the right of the point
        # print(value, val_error, split_precision)
        assert float(split_precision[0]) == 0.  # make sure the precision is < 1
        new_value = '{:.16f}'.format(value).split('.')  # put in float format
        new_number = [new_value[0], '.']  # get the > 1 value
        # print(new_value)
        decimal = [c for c in new_value[1]]   # line up the decimal values
        idx_digits = len(split_precision[1].rstrip('0'))  # get the index of the precision
        print(decimal)
        if int(decimal[idx_digits]) >= 5:  # round up
            error += 1
            # decimal[idx_digits-1] = str(int(decimal[idx_digits-1])+1)

        decimal[idx_digits] = '(%i)' % error   # replace the correct precision with the error

        
        new_number.extend(decimal[:idx_digits+1])  # build the list of the new number
        new_value = str(''.join(new_number))  # join
    
    # print('value: ', value, 'error: ', val_error, 'new_value: ', new_value)
    return new_value





def generate_latex_table(origin, target, keep_cols=[]):

    new_file = []
    with open(origin, "r") as file:
        for line in file:
            line = line.split(',')
            print(line)
            line = '\t\t &'.join(line[i].strip() for i in keep_cols).strip()
            # new_line = line.strip().replace(',', '&')
            # line += ' \\\\ \n'
            line += '\n'
            if len(line) > 0: new_file.append(line)
    
    with open(target, 'w') as f:
        for line in new_file:
            f.write(line)


def create_error_cols(origin, merge_cols, method=None, method_column=None):
    # create the error columns
    df = pd.read_csv(origin)

    for (x, _, _) in merge_cols:
        df[x] = ''

    rs_values = df['rs'].unique()
     
    if not method is None:
        new_data = []
        for rs in rs_values:
            for m in method[::-1]:
                if m in df[df['rs'] == rs][method_column].values:
                    break
            row = df[(df['rs'] == rs) & (df[method_column] == m)].values[0]
            new_data.append(row)

        df = pd.DataFrame(new_data, columns=df.columns)

    for i, row in enumerate(df.iterrows()):
        # print(i, row)
        for (new_col, old_col, old_err) in merge_cols:
            # print(new_col, old_col, old_err)
            err = row[1][old_err]
            # print(err)
            new_val = create_error_str_from_numbers(row[1][old_col], err)
            print(row[1]['rs'], new_val)
            df[new_col].iloc[i] = new_val

    df.to_csv(origin, index=False)

    return df


def saverio_2_csv(origin, target, save=True):
    new_file = []
    with open(origin, "r") as file:
        for line in file:
            new_line = line.split()
            if len(new_line) > 0: new_file.append(new_line)

    df = pd.DataFrame(data=new_file[1:], columns=new_file[0])

    dtypes = [float, str, float, float, float, float, float, float]
    for (k, v), dtype in zip(df.items(), dtypes):
        df[k] = df[k].astype(dtype)

    df.to_csv(target, index=False)
    
    # df = create_error_cols(df, merge_cols)

    return df

    


def load_pk(path):
        with open(path, 'rb') as f:
            x = pk.load(f)
        return x



def decorate(ax, title=None, xaxis=None, yaxis=None, xlims=None, ylims=None):
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if xlims is not None: ax.set_xlim(xlims) 
    if ylims is not None: ax.set_ylim(ylims)

# tricks with pandas
# df.loc[df['column name'] condition, 'new column name'] = 'value if condition is met'

def pretty_base(title=None, 
                xlabel=None, 
                ylabel=None, 
                xlims=None, 
                ylims=None,
                legend=None,
                xlines=[],
                ylines=[],
                latex=True,
                name=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman"})
        csfont = {}
        hfont = {}
    else:
        csfont = {'fontname':'Comic Sans MS'}
        hfont = {'fontname':'Helvetica'}

    ax.set_title(title, fontsize=18, **csfont)
    ax.set_xlabel(xlabel, fontsize=16, **hfont)
    ax.set_ylabel(ylabel, fontsize=16, **hfont)
    if legend is None:
        pass
    elif legend == 'outside':
        ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend == 'inside':
        ax.legend(fontsize=12)
    if xlims is not None: ax.set_xlim(xlims) 
    if ylims is not None: ax.set_ylim(ylims)
    ax.grid(b=True, which='major', axis='both')
    [ax.axvline(x=x, ls='--') for x in xlines]
    [ax.axhline(y=y, ls='--') for y in ylines]
    return fig, ax


def bokeh_bars(xs, ys, xerrs=None, yerrs=None):
    if xerrs is None:
        xlines = [(x, x) for x in xs]
    else:
        xlines = [(x-xerr, x+xerr) for x, xerr in zip(xs, xerrs)]
    if yerrs is None:
        ylines = [(y, y) for y in ys]
    else:
        ylines = [(y-yerr, y+yerr) for y, yerr in zip(ys, yerrs)]
    return xlines, ylines


def get_n_inputs(data):
    inacts = data['input_activation_nonlinearity'].values
    new_data = {'ncos': [], 'nsin': [], 'nkpoints': []}
    for nonlinearity in inacts:
        split = nonlinearity.split('+')
        nsin = ncos = nkpoints = 0
        if 'sin' in nonlinearity:
            sin_desc = [x for x in split if 'sin' in x][0]
            nsin = int(sin_desc[:-3]) if len(sin_desc) > 3 else 1

        if 'cos' in nonlinearity:
            cos_desc = [x for x in split if 'cos' in x][0]
            ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1

        if 'kpoints' in nonlinearity:
            kpoints_desc = [x for x in split if 'kpoints' in x][0]
            nkpoints = int(kpoints_desc[:-7])

        new_data['nsin'].append(nsin)
        new_data['ncos'].append(ncos)
        new_data['nkpoints'].append(nkpoints)

    data = data.assign(**new_data)
    return data


def get_data(target_dir, groupby=None, data_filename='config1.pk', dicts=[]):
    data_paths = find_all_files_in_dir(target_dir, data_filename)
    for data_path in data_paths:
        data = load_pk(data_path)
        dicts.append(data)
    df = pd.DataFrame(dicts)    # .filter(regex='energy')
    # columns = [x for x in df.columns if not ('dir' in x) and not ('path' in x)]
    # df = df[columns]
    
    try:
        columns = [c for c in df.columns if 'equilibrated_energy_mean_' in c]
        # df_tmp = df[columns].dropna()
        min_cols = df[columns].idxmin(axis=1, skipna=True)
        min_es, min_stds, min_sems, min_is = [], [], [], []
        for i, (row, min_col) in enumerate(zip(df.iterrows(), min_cols)):
            row = row[1]
            try:
                min_i = int(re.findall('\d+', min_col)[0])
                min_e = row[min_col]
                min_std = row['equilibrated_energy_std_i%i' % min_i]
                min_sem = row['equilibrated_energy_sem_i%i' % min_i]
                
            except Exception as e:
                print(e, min_col, row['name'])
                min_std = None
                min_sem = None
                min_e = None
                min_i = None
            min_stds.append(min_std)
            min_sems.append(min_sem)
            min_es.append(min_e)
            min_is.append(min_i)

        df['n_it_best_model'] = min_is
        df['equilibrated_energy_mean'] = min_es
        df['std_of_means'] = min_stds
        df['variance'] = df['std_of_means'] / np.sqrt(1000)
        df['equilibrated_energy_sem'] = min_sems
        df['energy (Ry)'] = df['equilibrated_energy_mean'] * 2.
        df = get_n_inputs(df)

        if groupby is not None:
            df = df.groupby(groupby).mean()
            df[groupby] = df.index
    except KeyError:
        print(df)

    return df


def plot_scatter(xs, 
              ys, 
              xlines=None,
              yerrs=None,
              xerrs=None,
              xticklabels=None,
              save_png=None,
              title='', 
              xlabel='', 
              ylabel='',
              xaxis='linear',
              yaxis='linear',
              hlines=None,
              graph=None,
              colors=None,
              line_label=None,
              legend_location='top_left'):

    if graph is None:
        graph = figure(title = title, 
                    x_axis_label=xlabel, 
                    y_axis_label=ylabel, 
                    width=400, height=400,
                    y_axis_type=yaxis, x_axis_type=xaxis)

    if colors is None:
        colors = itertools.cycle(palette) 

    if not xticklabels is None: 
        graph.xaxis.major_label_overrides = {i:name for i, name in enumerate(xticklabels)}
        graph.xaxis.ticker = xs
        graph.xaxis.major_label_orientation = 45

    if not type(xs) is list: xs = [xs]
    if not type(ys) is list: ys = [ys]
    if yerrs is None: yerrs = [None for _ in xs]
    elif not type(yerrs) is list: yerrs = [yerrs]

    for x, y, yerr in zip(xs, ys, yerrs):
        c = next(colors)
        if line_label is None:
            graph.circle(x, y, size=10, color=c)
        else:
            graph.circle(x, y, size=10, color=c, legend_label=line_label)
        if line_label is not None: graph.legend.location = legend_location
        if yerr is not None: graph.multi_line(*bokeh_bars(x, y, yerrs=yerr), color=c)

        if hlines is not None:
            for hline in hlines:
                graph.line(xs, [hline for _ in xs], line_width=2.)

        if save_png is not None:
            export_png(graph, filename = save_png)

    return graph



    # data_all = {'xs':[], 'ys':[], 'yerrs':[], 'xticklabels':[]}
    # for data_path in data_paths:
    #     data = load_pk(data_path)
    #     try:
    #         data_all['ys'].append(data[yname])
    #         if errname is not None: errs.append(data[errname])
    #         xs.append(data[hypam])
    #         appends.append([data_path[x] for x in append_xlabel])
    #     except Exception as e:
    #         print(e)
    #         continue

    # return data_all






def data_and_plot_scatter(target_dir, data_filename, 
              hypam=None, yname=None, errname=None,
              save_png=None, 
              append_xlabel=None,
              title='', xlabel='', ylabel='',
              hlines=None):
    data_paths = find_all_files_in_dir(target_dir, data_filename)
    

    

    colors = itertools.cycle(palette) 
    graph = figure(title = title, x_axis_label=xlabel, y_axis_label=ylabel, width=400, height=400)

    if type(xs[0]) is str:
        xticklabels = ['/'.join([x, *append]) for x, append in zip(xs, appends)]
        xs = [i for i in range(len(xs))]
        graph.xaxis.major_label_overrides = {i:name for i, name in enumerate(xticklabels)}
        graph.xaxis.ticker = xs

    elif type(xs[0]) is tuple:
        order = np.argsort([np.prod(x) for x in xs])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        xticklabels = [str(x) for x in xs]
        xs = [i for i in range(len(xs))]
        graph.xaxis.major_label_overrides = {i:name for i, name in enumerate(xticklabels)}
        graph.xaxis.ticker = xs

    c = next(colors)
    graph.circle(xs, ys, size=10, color=c)
    if errname is not None: graph.multi_line(*bokeh_bars(xs, ys, yerrs=errs), color=c)

    if hlines is not None:
        for hline in hlines:
            graph.line(xs, [hline for _ in xs], line_width=2.)

    if save_png is not None:
        export_png(graph, filename = save_png)
    show(graph)
    
    df = pd.DataFrame.from_dict(data={hypam:xticklabels, yname:ys, 'yerr': errs}, orient='columns')
    return df

# https://www.bing.com/search?q=matplotlib+list+font+families&cvid=0a051388b21c4469b7a31650ac135dfa&aqs=edge..69i57.4809j0j1&pglt=43&FORM=ANNTA1&PC=U531

if __name__ == '__main__':

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    if type(axs) is list:
        axs = [ax for lst in axs for ax in lst]

    path = ''

    # pandas
    data = pd.read_csv(path)  # header=None
    # data = data.sort_values(by=0)

    data = load_pk(path)

    axs.plot(X, Y, color='r', ls='--')
    axs.scatter(list1, list2, color='b', ms=10)

    axs.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle('')
    fig.tight_layout()

    plt.show()
    plt.savefig('junk.png')
