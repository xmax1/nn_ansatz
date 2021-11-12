


import argparse
from nn_ansatz import setup, run_vmc, approximate_energy

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, default='HEG')
    parser.add_argument("-nw", "--n_walkers", type=int, default=512)
    parser.add_argument("-n_sh", "--n_sh", type=int, default=64)
    parser.add_argument("-n_ph", '--n_ph', type=int, default=16)
    parser.add_argument('-orb', '--orbitals', type=str, default='real_plane_waves')
    parser.add_argument('-n_el', '--n_el', type=int, default=7)
    parser.add_argument('-inact', '--input_activation_nonlinearity', type=str, default='sin')
    parser.add_argument('-opt', '--opt', type=str, default='kfac')
    parser.add_argument('-n_det', '--n_det', type=int, default=1)
    parser.add_argument('-dp', '--density_parameter', type=float, default=1.0)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-name', '--name', type=str, default='junk')
    parser.add_argument('-n_it', '--n_it', type=int, default=10000)
    args = parser.parse_args()
    return args

args = get_args()

cfg = setup(**vars(args))

log = run_vmc(cfg)
approximate_energy(cfg, load_it=args.n_it)