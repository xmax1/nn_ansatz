
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pk
import scipy

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