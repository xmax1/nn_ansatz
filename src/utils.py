from gc import collect
import pickle as pk
import os
import yaml
import pandas as pd
import sys
from inspect import signature
from ast import literal_eval
from prettytable import PrettyTable
import numpy as np

oj = os.path.join


def ojm(*args):
    path = oj(*args)
    os.makedirs(path, exist_ok=True)
    return path


def save_pretty_table(df: pd.DataFrame, path='table.txt', cols=None, top=100, bottom=0.01):
    
    if cols is None: cols = df.columns

    table = PrettyTable()
    for c in cols:
        if df[c].dtypes not in [str, object]:
            if (top > df[c]).all() and (df[c] > bottom).all():
                df[c] = df[c].map(lambda x: '{:.3f}'.format(x))
            else:
                df[c] = df[c].map(lambda x: '{:.3e}'.format(x))
            
            table.add_column(c, list(df[c]))

    with open(path, 'w') as f:
        f.write(str(table))
    

def append_to_txt(path, data):
    with open(path, 'a') as f:
        f.write('\n ' + data)


def append_dict_to_dict(d, d_new):
    for k, v in d_new.items():
        if k not in d.keys():
            if isinstance(v, np.ndarray):
                d[k] = v
            else:
                d[k] = [v]
        else:
            if isinstance(v, np.ndarray):
                d[k] = np.concatenate([d[k], v])
            else:
                d[k].append(v)
    return d



def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x 


def load_yaml(path):
    with open(path) as f:
        sweep_cfg = yaml.safe_load(f)
    return sweep_cfg


def save_pk_and_csv(x, path):
    with open(path+'.pk', 'wb') as f:
        pk.dump(x, f)
    x = {k: ([v] if not isinstance(v, list) else v) for k, v in x.items()}  # needs at least one and values in lists
    df = pd.DataFrame.from_dict(x) 
    df.to_csv (path+'.csv', index = False, header=True)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def make_arg_key(key):
    key = key.replace('-', '')
    return key


def collect_args():
    if len(sys.argv) == 1:
        args = {}
    else:
        args = iter(sys.argv[1:])
        args = ' '.join(sys.argv[1:])
        args = args.split('--')[1:]  # first element is blank
        args = [a.split(' ', 1) for a in args]
        args = iter([x.replace(' ', '') for sub_list in args for x in sub_list])
        args = {make_arg_key(k):v for k, v in zip(args, args)}
    [print(k, v) for k, v in args.items()]
    return args


annotation_eval = [str,]


def type_args(args, fn):
    sig_params = signature(fn).parameters
    typed_args = {}
    for k, v in args.items():
        try:
            v = literal_eval(v)
        except Exception as e:
            print(v, str(v), e)
            v = str(v)

        # arg_type = sig_params[k].annotation
        # if arg_type in annotation_eval:
        #     v = arg_type(v)
        # else:
        #     v = literal_eval(v)

        typed_args[k] = v
    for k, v in typed_args.items():
        print(k, v, type(v))
    return typed_args




def run_fn_with_sysargs(fn):
    args = collect_args()
    typed_args = type_args(args, fn)
    fn(**typed_args)