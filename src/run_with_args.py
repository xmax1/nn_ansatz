


import argparse
from nn_ansatz import setup, run_vmc, approximate_energy
from distutils.util import strtobool

def input_bool(x):
    x = strtobool(x)
    if x: return True
    else: return False

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
    parser.add_argument('-name', '--name', type=str, default='junk')
    parser.add_argument('-n_it', '--n_it', type=int, default=10000)
    parser.add_argument('-act', '--nonlinearity', type=str, default='cos')
    parser.add_argument('-sim', '--simulation_cell', nargs='+', type=int, default=(1, 1, 1))
    parser.add_argument('-nl', '--n_layers', type=int, default=2)
    parser.add_argument('-npre', '--n_pre_it', type=int, default=500)
    parser.add_argument('--nopretrain',  dest='pretrain', action='store_false')
    parser.add_argument('--sweep', dest='sweep', action='store_true')
    parser.add_argument('-backflow_coords', default=True, type=input_bool)
    parser.add_argument('-jastrow', default=True, type=input_bool)
    parser.add_argument('-psplit_spins', default=True, type=input_bool)
    args = parser.parse_args()
    return args

args = get_args()

print(vars(args))
cfg = setup(**vars(args))

log = run_vmc(cfg)


if args.sweep == True:
    for load_it in range(10000, args.n_it+1, 10000):
        approximate_energy(cfg, load_it=load_it)
else:
    approximate_energy(cfg, load_it=args.n_it)