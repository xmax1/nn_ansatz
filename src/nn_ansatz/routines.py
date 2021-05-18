
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange


from .sampling import create_sampler
from .ansatz import create_wf
from .parameters import initialise_params, initialise_d0s, expand_d0s
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
# from .utils import *
from .optimisers import create_natural_gradients_fn, kfac
from .utils import Logging, load_pk, save_pk, key_gen, split_variables_for_pmap


def run_vmc(**cfg):

    logger = Logging(**cfg)

    key = rnd.PRNGKey(cfg['seed'])
    keys = rnd.split(key, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)

    wf, vwf, kfac_wf, wf_orbitals = create_wf(mol)
    params = initialise_params(key, mol)
    d0s = initialise_d0s(mol, cfg['n_devices'], cfg['n_walkers_per_device'])

    sampler, equilibrater = create_sampler(wf, vwf, mol, **cfg)

    walkers = None
    if cfg['load_pretrain']:
        params, walkers = load_pk(cfg['pre_path'])
    elif cfg['pretrain']:
        params, walkers = pretrain_wf(mol, **cfg)
    
    walkers = mol.initialise_walkers(walkers=walkers, **cfg)
    walkers = equilibrater(params, walkers, d0s, keys, n_it=1000, step_size=0.02)

    grad_fn = create_grad_function(wf, vwf, mol)

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(kfac_wf, wf, mol, params, walkers, d0s, **cfg)
    elif cfg['opt'] == 'adam':
        init, update, get_params = adam(cfg['lr'])
        update = jit(update)
        state = init(params)
    else:
        exit('Optimiser not available')

    steps = trange(0, cfg['n_it'], initial=0, total=cfg['n_it'], desc='training', disable=None)
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])

    for step in steps:
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, d0s, subkeys, step_size)

        grads, e_locs = grad_fn(params, walkers, d0s)

        if cfg['opt'] == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers, d0s)

        state = update(step, grads, state)
        params = get_params(state)

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance[0])


