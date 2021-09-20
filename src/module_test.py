from nn_ansatz import *

# confirm antisymmetric

import jax
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad, pmap
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange

from functools import partial
import itertools

from nn_ansatz import *

cfg = config = setup(system='LiSolidBCC',
               n_pre_it=0,
               n_walkers=64,
               n_layers=2,
               n_sh=32,
               n_ph=8,
               opt='kfac',
               n_det=2,
               print_every=1,
               save_every=5000,
               n_it=1000)

logger = Logging(**cfg)

keys = rnd.PRNGKey(cfg['seed'])
if bool(os.environ.get('DISTRIBUTE')) is True:
    keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

mol = SystemAnsatz(**cfg)

pwf = pmap(create_wf(mol), in_axes=(None, 0))
vwf = create_wf(mol)
swf = create_wf(mol, signed=True)

params = initialise_params(mol, keys)

sampler = create_sampler(mol, vwf)

ke = pmap(create_local_kinetic_energy(vwf), in_axes=(None, 0))
pe = pmap(create_potential_energy(mol), in_axes=(0, None, None))
grad_fn = create_grad_function(mol, vwf)

# walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=None)
# save_pk(walkers, 'walkers_no_infs.pk')
walkers = load_pk('walkers_no_infs.pk')

# using routines

def asym_test(swf, mol, params, walkers):
    up_idxs = range(0, mol.n_up)
    down_idxs = range(mol.n_up, mol.n_el)
    up_swaps = list(itertools.permutations(up_idxs, 2))
    down_swaps = list(itertools.permutations(down_idxs, 2))
    all_swaps = up_swaps + down_swaps

    awalkers = np.squeeze(np.array(walkers), axis=0)[0, ...][None, ...]
    bwalkers = awalkers.copy()
    for swap in all_swaps:
        bwalkers[0, swap, :] = awalkers[0, swap[::-1], :].copy()
        blog, bsign = swf(params, jnp.array(bwalkers))
        alog, asign = swf(params, jnp.array(awalkers))

        awalkers = bwalkers.copy()

        print('sign 1 %i sign 2 %i difference %.2f' % (asign, bsign, jnp.sum(blog - alog)))

# asym_test(swf, mol, params, walkers)


# log_psi = pwf(params, walkers)

# keys, subkeys = key_gen(keys)
# sampler(params, walkers, subkeys, mol.step_size)


# lr, damping, nc = 1e-4, 1e-4, 1e-4
# n_pre_it = 500
# n_walkers = 512
# n_layers = 2
# n_sh = 64
# n_ph = 16
# n_det = 8
# n_it = 1000
# seed = 1


# config = setup(system='LiSolidBCC',
#                n_pre_it=0,
#                n_walkers=64,
#                n_layers=2,
#                n_sh=32,
#                n_ph=8,
#                opt='kfac',
#                n_det=2,
#                print_every=1,
#                save_every=5000,
#                lr=lr,
#                n_it=1000,
#                norm_constraint=nc,
#                damping=damping)


# run_vmc(config)

