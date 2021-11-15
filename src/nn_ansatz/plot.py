
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pk
import scipy
from .utils import find_all_files_in_dir
from bokeh.io import output_notebook, export_png
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
import itertools

# params = {'legend.fontsize': 16,
#           'legend.handlelength': 3}

# plt.rcParams.update(params)

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
    elif legend is 'outside':
        ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend is 'inside':
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


def get_data(target_dir, data_filename='config1.pk', dicts=[]):
    data_paths = find_all_files_in_dir(target_dir, data_filename)
    for data_path in data_paths:
        data = load_pk(data_path)
        dicts.append(data)
    df = pd.DataFrame(dicts)    # .filter(regex='energy')
    columns = [x for x in df.columns if not ('dir' in x) and not ('path' in x)]
    
    df = df[columns]
    return df


def plot_scatter(xs, 
              ys, 
              yerrs=None,
              xerrs=None,
              xticklabels=None,
              save_png=None,
              title='', 
              xlabel='', 
              ylabel='',
              hlines=None):

    colors = itertools.cycle(palette) 
    graph = figure(title = title, x_axis_label=xlabel, y_axis_label=ylabel, width=400, height=400)

    if not xticklabels is None: 
        graph.xaxis.major_label_overrides = {i:name for i, name in enumerate(xticklabels)}
        graph.xaxis.ticker = xs
        graph.xaxis.major_label_orientation = 45

    # elif type(xs[0]) is tuple:
    #     order = np.argsort([np.prod(x) for x in xs])
    #     xs = [xs[i] for i in order]
    #     ys = [ys[i] for i in order]
    #     xticklabels = [str(x) for x in xs]
    #     xs = [i for i in range(len(xs))]
    #     graph.xaxis.major_label_overrides = {i:name for i, name in enumerate(xticklabels)}
    #     graph.xaxis.ticker = xs

    c = next(colors)
    graph.circle(xs, ys, size=10, color=c)
    if yerrs is not None: graph.multi_line(*bokeh_bars(xs, ys, yerrs=yerrs), color=c)

    if hlines is not None:
        for hline in hlines:
            graph.line(xs, [hline for _ in xs], line_width=2.)

    if save_png is not None:
        export_png(graph, filename = save_png)
    show(graph)



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