import os

import jax
from jax._src.tree_util import tree_flatten
import jax.random as rnd
import jax.numpy as jnp
from jax import jit, pmap
from jax.experimental.optimizers import adam
from tqdm import trange
import sys

from .ansatz_base import split_and_squeeze

from .python_helpers import save_pk

from .sampling import create_sampler, initialise_walkers
from .ansatz import create_wf
from .parameters import initialise_params
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
from .optimisers import kfac
from .utils import Logging, compare, load_pk, key_gen, split_variables_for_pmap, write_summary_to_cfg


def check_inf_nan(arg):
    return jnp.isnan(arg).any() or jnp.isinf(arg).any()


def check_and_save(args, names):
    for arg in args:
        check = check_inf_nan(arg)
        if check:
            for arg, name in zip(args, names):
                save_pk(arg, '%s.pk' % name)
            sys.exit('found nans')

def run_vmc(cfg, walkers=None):

    logger = Logging(**cfg)

    keys = rnd.PRNGKey(cfg['seed'])
    if bool(os.environ.get('DISTRIBUTE')) is True:
        keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)

    vwf = create_wf(mol)
    params = initialise_params(mol, keys)

    sampler = create_sampler(mol, vwf)

    if cfg['load_pretrain']:
        params, walkers = load_pk(cfg['pre_path'])
    elif cfg['pretrain']:
        params, walkers = pretrain_wf(mol, **cfg)
    
    if walkers is None:
        walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=walkers)

    grad_fn = create_grad_function(mol, vwf)

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])
    elif cfg['opt'] == 'adam':
        init, update, get_params = adam(cfg['lr'])
        update = jit(update)
        state = init(params)
    else:
        exit('Optimiser not available')

    steps = trange(1, cfg['n_it']+1, initial=1, total=cfg['n_it']+1, desc='training', disable=None)
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])

    for step in steps:
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        
        grads, e_locs = grad_fn(params, walkers)

        if cfg['opt'] == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers)

        state = update(step, grads, state)
        params = get_params(state)
        
        check_and_save([walkers, e_locs], ['walkers', 'e_locs'])

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance)

    write_summary_to_cfg(cfg["csv_cfg_path"], logger.summary)
    logger.walkers = walkers
    
    return logger


def compare_einsum(cfg, walkers=None):
    logger = Logging(**cfg)

    keys = rnd.PRNGKey(cfg['seed'])
    if bool(os.environ.get('DISTRIBUTE')) is True:
        keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)
    mol_einsum = SystemAnsatz(**cfg)

    vwf = create_wf(mol, distribute=True)
    params = initialise_params(mol, keys)

    sampler = create_sampler(mol, vwf)
    
    mol_einsum.einsum = True
    vwf_einsum = create_wf(mol_einsum, distribute=True)
    params_einsum = initialise_params(mol_einsum, keys)
    
    compare_params(params, params_einsum)
        
    if walkers is None:
        walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=walkers)

    vwf = pmap(vwf, in_axes=(None, 0))
    vwf_einsum = pmap(vwf_einsum, in_axes=(None, 0))
    log_psi = vwf(params, walkers)
    log_psi_einsum = vwf_einsum(params_einsum, walkers)

    difference = jnp.mean(log_psi - log_psi_einsum)
    print('difference = %.2f' % difference)


def equilibrate(params, walkers, keys, mol=None, vwf=None, sampler=None, compute_energy=None, n_it=1000, step_size=0.02):
    
    if sampler is None:
        if vwf is None:
            vwf = create_wf(mol)
        sampler = create_sampler(vwf, mol)
    if compute_energy is None:
        if vwf is None:
            vwf = create_wf(mol)
        compute_energy = create_energy_fn(vwf, mol)

    step_size = split_variables_for_pmap(walkers.shape[0], step_size)

    for i in range(n_it):
        keys, subkeys = key_gen(keys)

        walkers, acc, step_size = sampler(params, walkers, subkeys, step_size)
        e_locs = compute_energy(params, walkers)
        e_locs = jax.device_put(e_locs, jax.devices()[0]).mean()
        print('step %i energy %.4f acceptance %.2f' % (i, jnp.mean(e_locs), acc[0]))

    return walkers


def compare_params(params, params_einsum):
    pi_ups = [split_and_squeeze(p, axis=2) for p in split_and_squeeze(params_einsum['env_pi_up'], axis=2)]
    pi_ups = [x for y in pi_ups for x in y]

    pi_downs = [split_and_squeeze(p, axis=2) for p in split_and_squeeze(params_einsum['env_pi_down'], axis=2)]
    pi_downs = [x for y in pi_downs for x in y]

    for dictionary, name in zip([pi_ups, pi_downs], ['env_pi_up', 'env_pi_down']):
        for i, v in enumerate(dictionary):
            params_einsum[name + '_%i' %i] = v
    
    params_einsum = {k:v for k, v in params_einsum.items() if not 'env_pi_up' == k}
    params_einsum = {k:v for k, v in params_einsum.items() if not 'env_pi_down' == k}

    for (k1, v1), (k2, v2) in zip(params.items(), params_einsum.items()):
        if not v1.shape == v2.shape:
            print(v1.shape, v2.shape)
            v2 = v2.reshape(v1.shape)
        print(jnp.mean(v1 - v2))


if __name__ == '__main__':
    import sys
    sys.path.append('.')

    from utils import setup
    cfg = setup(system='LiSolidBCC',
                    n_pre_it=0,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=64,
                    step_size=0.05,
                    n_ph=32,
                    scalar_inputs=False,
                    orbitals='anisotropic',
                    einsum=True,
                    n_periodic_input=1,
                    opt='adam',
                    n_det=4,
                    print_every=50,
                    save_every=2500,
                    lr=1e-4,
                    n_it=30000)

    compare_einsum(cfg)

