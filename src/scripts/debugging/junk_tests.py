
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
def pretty_base(title=None, 
                xaxis=None, 
                yaxis=None, 
                xlims=None, 
                ylims=None,
                legend=None,
                figsize=(4,4),
                xlines=[],
                ylines=[]):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    csfont = {'fontname':'Times New Roman'}
    hfont = {'fontname':'Helvetica'}

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xaxis, fontsize=14)
    ax.set_ylabel(yaxis, fontsize=14)
    if legend is None:
        pass
    elif legend is 'outside':
        ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend is 'inside':
        ax.legend(fontsize=12)
    if xlims is not None: ax.set_xlim(xlims) 
    if ylims is not None: ax.set_ylim(ylims)
    ax.grid(b=True, which='major', axis='both')
    [ax.axhline(y=x, ls='--') for x in xlines]
    [ax.axvline(x=y, ls='--', alpha=0.5, color='r') for y in ylines]
    return fig, ax

# fig, ax = plt.subplots(1, 1)

pretty_base(title='x')
plt.savefig('x.png')