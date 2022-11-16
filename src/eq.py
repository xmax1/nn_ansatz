
import sys
from time import time
from typing import Callable
from itertools import product
from math import ceil, log


import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as rnd

from pathlib import Path

from utils import load_pk
from nn_ansatz.systems import SystemAnsatz
from nn_ansatz.ansatz import create_wf
from nn_ansatz.sampling import create_sampler, generate_walkers
from functools import partial
from walle_utils import StatsCollector, save_pk
from nn_ansatz.vmc import create_energy_fn



def equilibrate_1M_walkers(
    key, 
    n_walkers_max, 
    walkers, 
    n_warm_up_steps=1000000,
    n_walkers_eq_max=256, 
    e_every=100, 
    step_size=0.02,
    n_eq_walkers=1000000,
):
    
    walkers = jnp.array(load_pk('/home/energy/amawi/projects/nn_ansatz/src/experiments/PRX_Responses/step100_100rs_trw.pk'))
    print('WARMING UP EQ')
    t0 = time()
    for step in range(n_warm_up_steps):
        key, subkey = rnd.split(key, 2)
        walkers, acc, step_size = sampler(params, walkers, subkey, step_size)
        
        if step % e_every == 0:
            pe, ke = compute_energy(params, walkers)
            print(f'Step: {step} | E: {jnp.mean(pe+ke):.3f} +- {jnp.std(pe+ke):.3f}')
            print(f'Step_size: {float(step_size):.3f} | Acc.: {float(acc):.2f}')
        
        t0 = track_time(step, n_warm_up_steps, t0, every_n=e_every)

    save_pk(np.array(walkers), '/home/energy/amawi/projects/nn_ansatz/src/experiments/PRX_Responses/step100_100rs_trw_then_1MWarmUp.pk')

    print('EQUILIBRATION LOOP')
    n_batch = int((n_eq_walkers//n_walkers_max))

    ex_stats = StatsCollector()
    for step in range(1, n_batch+1):
        key, subkey = rnd.split(key, 2)

        idxs_all = jnp.arange(0, len(walkers))
        idxs = rnd.choice(subkey, idxs_all, (n_walkers_max, ))
        w = walkers[idxs]

        w, acc, step_size = sampler(params, w, subkey, step_size)
        
        walkers = jnp.concatenate([walkers, w], axis=0)
        
        if step % e_every == 0:
            walkers_tmp = w[:n_walkers_eq_max]
            pe, ke = compute_energy(params, walkers_tmp)
            ex_stats.ke = jnp.mean(ke)
            ex_stats.pe = jnp.mean(pe)
            ex_stats.e = jnp.mean(ke + pe)
            print(f'Step: {step} | E: {jnp.mean(ke + pe):.6f} | n_eq: {len(walkers)}')
            print(f'Step_size: {float(step_size):.3f} | Acc.: {float(acc):.2f}')
            print(f'idxs: {jnp.mean(idxs):.3f}')

    save_pk(ex_stats.get_dict(), results_dir / f'e_stats_i{load_it}.pk')
    return walkers



def transform_vector_space(vectors: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis


def keep_in_boundary(walkers, basis, inv_basis):
    talkers = transform_vector_space(walkers, inv_basis)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0., talkers + 1., talkers)
    talkers = transform_vector_space(talkers, basis)
    return talkers


def make_arg_key(key):
    key = key.replace('-', '')
    return key


def collect_args():
    if len(sys.argv) == 1:
        args = {}
    else:
        args = ' '.join(sys.argv[1:])
        args = args.split('--')[1:]  # first element is blank
        args = [a.split(' ', 1) for a in args]
        args = iter([x.replace(' ', '') for sub_list in args for x in sub_list])
        args = {make_arg_key(k):v for k, v in zip(args, args)}
    return args



def track_time(
    step: int, 
    total_step: int, 
    t0: float, 
    every_n: int=10,
    tag: str=''
):
    """ Prints time
    
    """
    if step % every_n == 0:
        d, t0 = time() - t0, time()
        print(
            f'{tag} | \
            step {step} % complete: {step/float(total_step):.2f} | \
            t: {(d):.2f} |'
        )
    return t0  # replaces t0 external

closest = lambda x: int(ceil(log(x) / log(2)))

def input_bool(x):
    from distutils.util import strtobool
    x = strtobool(x)
    if x: return True
    else: return False

args = collect_args()

print(run_dir := Path(args['run_dir']))
print(n_walkers_max := int(args['n_walkers_max']))
print(equilibrate := input_bool(args['equilibrate']))
print(n_walkers := int(args['n_walkers']))
print(n_average_sphere := int(args['n_average_sphere']))
print(n_points := int(args['n_points']))
print(n_dim_exp := int(args['n_dim_exp']))
print(n_points_hypercube := int(args['n_points_hypercube']))
print(load_it := int(args['load_it']))
n_k_basis = 2

results_dir = Path('/scratch/amawi', run_dir)

cfg = load_pk(run_dir / 'config1.pk')
for k, v in cfg.items():
    if not isinstance(v, str):
        if 'equilibrate' not in k or 'equilibrated_energy_mean' in k:
            print(k, '\n', v)

n_el, n_up, seed = cfg['n_el'], cfg['n_up'], cfg['seed']
key = rnd.PRNGKey(seed)

models_path = run_dir / 'models'
walkers_dir = run_dir / 'walkers'
load_file = f'i{load_it}.pk'

mol = SystemAnsatz(**cfg)
vwf = jit(create_wf(mol))
swf = jit(create_wf(mol, signed=True))
sampler = create_sampler(mol, vwf)
params = load_pk(models_path / f'i{load_it}.pk')

compute_energy = create_energy_fn(mol, vwf, separate=True)

shift = mol.scale_cell / 3.
sites = np.linspace(0, mol.scale_cell, 4, endpoint=True)[:-1] + (shift/2.)
print('sites', sites)
sites_all = np.array(list(product(*[sites, sites, sites])))
walkers = []
for i in range(n_walkers_max):
    idxs = np.random.choice(np.arange(len(sites_all)), size=n_el, replace=False)
    sites = sites_all[idxs]   
    walkers.append(sites)
walkers = jnp.array(np.stack(walkers, axis=0))

print('WARMING UP EQ')
t0 = time()
n_warm_up_steps = 1000
e_every = 10
step_size = 0.05
for step in range(n_warm_up_steps):
    key, subkey = rnd.split(key, 2)
    walkers, acc, step_size = sampler(params, walkers, subkey, step_size)
    if step % e_every == 0:
        pe, ke = compute_energy(params, walkers)
        e = float(jnp.mean(pe+ke))
        print(f'Step: {step} | E: {e:.3f} +- {jnp.std(pe+ke):.3f}')
        print(f'Step_size: {float(step_size):.3f} | Acc.: {float(acc):.2f}')
        t0 = track_time(step, n_warm_up_steps, t0, every_n=e_every)

e = str(e)[:8]
w_path = f'/home/energy/amawi/projects/nn_ansatz/src/experiments/PRX_Responses/runs/run301115/w_e{e}_i{load_it}.pk'
save_pk(np.array(walkers), w_path)
