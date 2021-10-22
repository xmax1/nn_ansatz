import os

import jax
import jax.random as rnd
import jax.numpy as jnp
from jax import jit
from jax.experimental.optimizers import adam
from tqdm import trange

from .sampling import create_sampler, initialise_walkers
from .ansatz import create_wf
from .parameters import initialise_params
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
from .optimisers import kfac
from .utils import Logging, load_pk, key_gen, split_variables_for_pmap, write_summary_to_cfg


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

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance[0])

    write_summary_to_cfg(cfg["csv_cfg_path"], logger.summary)
    logger.walkers = walkers
    
    return logger


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