import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad, pmap
from jax.experimental.optimizers import adam
from jax.tree_util import tree_flatten, tree_unflatten
from jax import tree_util
from tqdm.notebook import trange

from src.nn_ansatz import *

import os
os.environ['JAX_PLATFORM_NAME']='cpu'
# os.environ['no_JIT'] = 'True'
# os.environ['XLA_FLAGS']="--xla_force_host_platform_device_count=4"
# os.environ['XLA_FLAGS']="--xla_dump_to=/tmp/foo"

import jax
print(jax.devices())

import numpy as np
from functools import partial

import jax.numpy as jnp

lr, damping, nc = 1e-4, 1e-4, 1e-4

cfg = setup(system='LiSolid',
               n_pre_it=501,
               n_walkers=8,
               n_layers=2,
               n_sh=16,
               n_ph=4,
               opt='kfac',
               n_det=2,
               print_every=1,
               save_every=5000,
               lr=1e-3,
               n_it=1000,
               norm_constraint=1e-4,
               damping=1e-3,
               kappa = 0.5,
               real_cut = 5,
               reciprocal_cut = 5)

key = rnd.PRNGKey(cfg['seed'])
keys = rnd.split(key, cfg['n_devices']).reshape(cfg['n_devices'], 2)

mol = SystemAnsatz(**cfg)
params = initialise_params(mol, key)



vwalkers = generate_walkers_around_nuclei(mol.n_el_atoms, mol.atom_positions, mol.n_walkers)
pwalkers = vwalkers[None, ...]

# partial it before the vmap  # need to partial the d0s before vmapping
vwf = create_wf(mol)
vwf(params, vwalkers)

sampler = create_sampler(mol, vwf)
walkers = initialise_walkers(mol, vwf, sampler, params, keys)

pwf = pmap(vwf, in_axes=(None, 0))
fwd = pwf(params, walkers)

keys, subkeys = key_gen(keys)
sampler(params, walkers, keys, mol.step_size)

# grad_fn = create_grad_function(wf, vwf, mol)

kinetic = create_local_kinetic_energy(vwf)
e = kinetic(params, vwalkers)
pkinetic = pmap(kinetic, in_axes=(None, 0))
e = pkinetic(params, walkers)

compute_energy = pmap(create_energy_fn(vwf, mol), in_axes=(None, 0))  # ctrl shift space

# e_locs = compute_energy(params, walkers)

# print(e_locs)









# if cfg['opt'] == 'kfac':
#     update, get_params, kfac_update, state = kfac(kfac_wf, wf, mol, params, walkers, d0s, **cfg)
# elif cfg['opt'] == 'adam':
#     init, update, get_params = adam(cfg['lr'])
#     update = jit(update)
#     state = init(params)
# else:
#     exit('Optimiser not available')

# steps = trange(0, cfg['n_it'], initial=0, total=cfg['n_it'], desc='training', disable=None)
# step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])

# for step in steps:
#     keys, subkeys = key_gen(keys)

#     walkers, acceptance, step_size = sampler(params, walkers, d0s, subkeys, step_size)

#     grads, e_locs = grad_fn(params, walkers, d0s)

#     if cfg['opt'] == 'kfac':
#         grads, state = kfac_update(step, grads, state, walkers, d0s)

#     state = update(step, grads, state)
#     params = get_params(state)

#     steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
#     steps.refresh()

#     logger.log(step,
#                 opt_state=state,
#                 params=params,
#                 e_locs=e_locs,
#                 acceptance=acceptance[0])



