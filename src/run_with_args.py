


import argparse
from nn_ansatz import setup, run_vmc, approximate_energy
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
    parser.add_argument('-save_every', '--save_every', default=10000, type=int)
    parser.add_argument('-bf_af', '--bf_af', default='cos', type=str)
    parser.add_argument('-run_dir', '--run_dir', default=None, type=str)
    parser.add_argument('-out_dir', '--out_dir', default='', type=str)
    args = parser.parse_args()
    return args

args = get_args()

print(vars(args))
cfg = setup(**vars(args))

log = run_vmc(cfg)

if args.sweep == True:
    for load_it in range(args.save_every, args.n_it+1, args.save_every):
        approximate_energy(cfg, run_dir=args.run_dir, load_it=load_it, n_it=20000)
else:
    cfg = approximate_energy(cfg, run_dir=args.run_dir, load_it=args.n_it, n_it=20000)

new_cfg = {}
for k, v in cfg.items():
    if isinstance(v, jnp.ndarray):
        v = np.array(v)
    new_cfg[k] = v

uppers = string.ascii_uppercase
lowers = string.ascii_lowercase
numbers = ''.join([str(i) for i in range(10)])
characters = uppers + lowers + numbers
name = ''.join([random.choice(characters) for i in range(20)]) + '.pk'

print('SAVING THE FUCKING FILE')

folder = f'/home/energy/amawi/projects/nn_ansatz/src/experiments/{args.exp_name}/results'
path = Path(folder + f'/{name}')
mkdir(path)
save_pk(new_cfg, path)



