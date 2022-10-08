from difflib import get_close_matches
from typing import Any, Iterable, Callable
import numpy as np
import pickle as pk

from pathlib import Path
import datetime
import os

from matplotlib import pyplot as plt

oj = Path

def format_ax(
    ax,
    
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    
    xticks: list | None = None,
    xticklabels: list | None = None,
    
    yticks: list | None = None,
    yticklabels: list | None = None, 

    hlines: float | list | None = None,
    hlines_kwargs: dict | None = None,

    vlines: float | list | None = None,
    vlines_kwargs: dict | None = None,

    legend: bool = False,
):  

    def get_font():
        import sys
        from pathlib import Path
        import os
        import site
        from difflib import get_close_matches
        import os

        sp_path = Path(site.getsitepackages()[0]) / 'matplotlib/mpl-data/fonts'
        all_files = [f for f in sp_path.rglob("*") if '.ttf' in f.name]

        # ~/.conda/envs/td/lib/python3.10/site-packages/matplotlib/mpl-data

        fontname = font_manager.FontProperties(fname='/project/phusers/test/fonts/edukai-4.0.ttf')
        return

    plt.style.use("tableau-colorblind10")
    # plt.style.available
    # print(*font_manager.findSystemFonts(fontpaths=None, fontext='ttf'), sep="\n")

    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation = 45, ha="right")
    else:
        ax.ticklabel_format(
            axis='x', 
            style='sci', 
            scilimits=(-2, 3), 
            useMathText=True, 
            useOffset=True
        )

    if yticklabels is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation = 45, ha="right")
    else:
        ax.ticklabel_format(
            axis='y', 
            style='sci', 
            scilimits=(-2, 3), 
            useMathText=True, 
            useOffset=True
        )


    

    ax.tick_params(axis='both', 
                   pad=3.)  # pad default is 4.
    
    default = {'color': 'red'}

    if hlines is not None:
        args = default | hlines_kwargs
        ax.hlines(hlines, *xlim, **args)

    if vlines is not None:
        args = default | vlines_kwargs
        ax.vlines(vlines, *ylim, **args)

    if legend: 
        ax.legend()

    if False:
        ax.tick_params(
            axis='both',
            rotation=0.,
            pad=10.
        )
    return ax


def append_idx(root, suffix: str | None = None):
    idx = 0
    root_tmp = root + f'_{idx}' + suffix
    while os.path.exists(root_tmp):
        root_tmp = root + f'_{idx}' + suffix
        idx += 1
    return root_tmp


def format_fig(
        fig,
        axs,
        fig_title: str | None = None,
        tight_layout: bool = True,
        fig_path: str | None = None,
    ):

    [ax.remove() for ax in axs]

    fig.suptitle(fig_title)
    fig.tight_layout()
    if fig_path is None: 
        fig_path = ojm('./throwaway', datetime.now().strftime("%d%b"), 'exp.png')
        ojm(fig)
        fig_path = append_idx(fig_path)
    fig.savefig(fig_path)



def ojm(*args):
    '''
    star operator here turns it into a tuple even if single element, 
    therefore this works for just making dirs
    
    takes a path
    removes the filename if filepath 
    creates the directory if it doesn't exist 
    returns the whole path 
    '''
    path = Path(*args)
    if path.suffix != '':
        root = path.parent
    else:
        root = path
    root.mkdir(parents=True, exist_ok=True)
    return path

def save_pk(data: dict | np.ndarray, path: str):
    with open(path, 'wb') as f:
        pk.dump(data, f)


def save_dict_as_yaml(d: dict, path: str):
    ''' like this to allow recursion '''
    with open(path, 'w') as f:
        write_dict(d, f)


class BaseGet():
    
    def get(self, names: str | list, alternate: str | None = None) -> Any:
        
        if isinstance(names, str):
            names = self.check_and_fudge_key(names)
            return self.__dict__.get(names, alternate)

        new_names = []
        for name in names:
            name = self.check_and_fudge_key(name)
            new_names.append(name)

        return [self.__dict__.get(name, alternate) for name in new_names]

    def get_dict(self):
        return self.__dict__

    def __getattr__(self, __key: str) -> Any:
        __key = self.check_and_fudge_key(__key)
        return self.__dict__.get(__key)

    def check_and_fudge_key(self, key):
        keys = self.__dict__.keys()
        if key not in keys:
            print(f'Finding closest match for name {key} getter, maybe you have a lil buggy bug boop')
            matches = get_close_matches(key, keys, n=1, cutoff=0.5)
            if len(matches) > 0:
                match = matches[0]
                print(f'Guessing {match} for key attempt {key}')
            else:
                match = 'THIS_KEY_DOES_NOT_EXIST'
            return match
        return key



class StatsCollector(BaseGet):
    def __init__(self):
        super().__init__()

    def __setattr__(self, __name: str, __value: int | float | Iterable) -> None:
        
        __value = np.array(__value)
        
        if __value.ndim == 0: 
            __value = __value[None]

        if __name not in self.__dict__.keys():
            self.__dict__[__name] = __value
        else:
            self.__dict__[__name] = np.concatenate([self.__dict__[__name], __value], axis=0)

    def set_by_name(self, k, v):
        self.__setattr__(k, v)

    def process(self, names: str | list, fn: Callable):
        
        if isinstance(names, str):
            self.__dict__[names] = np.array(fn(self.__dict__[names]))
        else:
            for name in names:
                self.__dict__[name] = np.array(fn(self.__dict__[name]))

    def overwrite(self, k, v):
        self.__dict__[k] = np.array(v)


class DictToClass(BaseGet):
    
    def __init__(self, d: dict) -> None:
        super().__init__()
        
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def save(self):
        
        save_pk(self.get_dict(), self.path.with_suffix('.pk'))
        save_dict_as_yaml(self.get_dict(), self.path.with_suffix('.yaml'))

    def merge(self, 
              d_new: dict, 
              tag: str | None = None, 
              overwrite: bool = False, 
              only_matching: bool = False):
        
        if not overwrite:
            print(f'Not updating {[k for k in d_new.keys() if k not in self.keys()]}')
            d_new_filtered = {k:v for k,v in d_new.items() if k not in self.keys()}
        
        if only_matching:
            d_new_filtered = {k:v for k,v in d_new_filtered.items() if k in self.keys()}
        
        for k, v in d_new_filtered.items():
            setattr(self, k, v)
        
        if not tag is None:
            setattr(self, tag, d_new)