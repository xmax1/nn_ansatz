import jax
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad, pmap
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange

from functools import partial

import os

from nn_ansatz import *


cfg = setup(system='HEG',
            n_walkers=512,
            n_layers=2,
            n_sh=64,
            step_size=0.05,
            n_ph=32,
            orbitals='real_plane_waves',
            n_el = 7,
            input_activation_nonlinearity='sin+cos+bowl',
            n_periodic_input=1,
            opt='kfac',
            n_det=1,
            density_parameter=1.,
            lr = 1e-4,
            n_it=1000,
            name='111121/pre_test')


mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, None)
params, walkers = pretrain_wf(mol, params, keys, sampler, walkers, n_eq_it=100)





### ENERGY DEBUG
# cfg = load_pk('/home/amawi/projects/nn_ansatz/src/exp/experiments/HEG/101121/heg_test/kfac_1lr-4_1d-3_1nc-4_m512_s64_p32_l2_det4/run1/config1.pk')
# # os.environ['DISTRIBUTE'] =  'no'
# walkers = None

# mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)
# walkers = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=True, n_it=200)
# energy_function = create_energy_fn(mol, vwf, separate=True)

# if bool(os.environ.get('DISTRIBUTE')) is True:
#     energy_function = pmap(energy_function, in_axes=(None, 0))

# local_kinetic_energy = create_local_kinetic_energy(vwf)
# if bool(os.environ.get('DISTRIBUTE')) is True:
#     local_kinetic_energy = pmap(local_kinetic_energy, in_axes=(None, 0))

# ke = local_kinetic_energy(params, walkers)