


import argparse
from nn_ansatz import setup, run_vmc
from nn_ansatz.routines import compute_energy_from_save
from distutils.util import strtobool
import numpy as np
from jax import numpy as jnp
import time
import string
import pickle as pk
import numpy as np
import random
import os
from pathlib import Path

def input_bool(x):
    from distutils.util import strtobool
    x = strtobool(x)
    if x: return True
    else: return False

def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)

def mkdir(path: Path):
    path = Path(path)
    if path.suffix != '':
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, default='HEG')
    parser.add_argument("-nw", "--n_walkers", type=int, default=1024)
    parser.add_argument("-n_sh", "--n_sh", type=int, default=128)
    parser.add_argument("-n_ph", '--n_ph', type=int, default=32)
    parser.add_argument('-orb', '--orbitals', type=str, default='real_plane_waves')
    parser.add_argument('-n_el', '--n_el', type=int, default=7)
    parser.add_argument('-n_up', '--n_up', type=int, default=None)
    parser.add_argument('-inact', '--input_activation_nonlinearity', type=str, default='3sin+3cos')
    parser.add_argument('-opt', '--opt', type=str, default='kfac')
    parser.add_argument('-n_det', '--n_det', type=int, default=1)
    parser.add_argument('-dp', '--density_parameter', type=float, default=1.0)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-atol', '--atol', type=float, default=1e-6)
    parser.add_argument('-n_it', '--n_it', type=int, default=10000)
    parser.add_argument('-act', '--nonlinearity', type=str, default='cos')
    parser.add_argument('-sim', '--simulation_cell', nargs='+', type=int, default=(1, 1, 1))
    parser.add_argument('-nl', '--n_layers', type=int, default=2)
    parser.add_argument('-cl', '--correlation_length', type=int, default=10)
    parser.add_argument('-npre', '--n_pre_it', type=int, default=500)
    parser.add_argument('-pretrain',  '--pretrain', default=True, type=input_bool)
    parser.add_argument('-sweep', '--sweep', default=False, type=input_bool)
    parser.add_argument('-backflow_coords', '--backflow_coords', default=True, type=input_bool)
    parser.add_argument('-jastrow', '--jastrow', default=True, type=input_bool)
    parser.add_argument('-psplit_spins', '--psplit_spins', default=True, type=input_bool)
    parser.add_argument('-ta', '--target_acceptance', default=0.5, type=float)
    parser.add_argument('-seed', '--seed', default=0, type=int)
    parser.add_argument('-exp_name', '--exp_name', default=None, type=str)
    parser.add_argument('-save_every', '--save_every', default=5000, type=int)
    parser.add_argument('-bf_af', '--bf_af', default='no_af', type=str)
    parser.add_argument('-run_dir', '--run_dir', default=None, type=str)
    parser.add_argument('-out_dir', '--out_dir', default='', type=str)
    parser.add_argument('-final_sprint', '--final_sprint', default=0.05, type=float)
    parser.add_argument('-n_walkers_per_device', '--n_walkers_per_device', default=256, type=int)
    parser.add_argument('-n_compute', '--n_compute', default=1000, type=int)
    parser.add_argument('-load_it', '--load_it', default=0, type=int)
    parser.add_argument('-do', '--do', default='vmc', type=str)
    args = parser.parse_args()
    # if you add a key ADD IT HERE AND IN UTILS
    return args

args = get_args()
cfg = vars(args)

print('PRE SETUP')
for k, v in cfg.items():
    print(k, '\n', v)

cfg = setup(**cfg)

print('POST SETUP')
for k, v in cfg.items():
    print(k, '\n', v)

if cfg["do"] == 'vmc':
    print(f'RUNNING {cfg["do"]}')
    
    log = run_vmc(cfg)
elif cfg["do"] == 'compute_e':
    print(f'RUNNING {cfg["do"]}')
    compute_energy_from_save(cfg)



