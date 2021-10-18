import os
import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')

import jax
from jax import pmap, vmap
from jax.tree_util import tree_flatten
from jax.experimental.optimizers import adam
import jax.numpy as jnp
from jax import lax

from functools import partial
import numpy as np
from tqdm.notebook import trange

import matplotlib.pyplot as plt
from matplotlib import cm

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

# pwf = pmap(create_wf(mol), in_axes=(None, 0))
vwf = create_wf(mol)
# jswf = jit(create_wf(mol, signed=True))

# sampler = create_sampler(mol, vwf)

# params = initialise_params(mol, keys)
params = load_pk('params.pk')
d0s = initialise_d0s(mol)

# walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=None)
# save_pk(walkers, 'walkers_no_infs.pk')
walkers = load_pk('walkers_no_infs.pk')

ke = create_local_kinetic_energy(vwf)
pe = create_potential_energy(mol)


def break_if_nan(tensor):
    if jnp.isnan(tensor).any():
        print(tensor)
        exit()


def norm(x, axis=-1):
    y = x**2
    z = jnp.sum(y, axis=axis)
    return jnp.sqrt(z)



def anisotropic_exponent(ae_vector, sigma, d0, n_spin, unit_cell_length=1., periodic_boundaries=False):
    # sigma (3, 3 * n_det * n_spin)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    
    if periodic_boundaries:

        ae_vector = jnp.where(ae_vector < -0.25 * unit_cell_length, -unit_cell_length**2/(8.*(unit_cell_length + 2.*ae_vector)), ae_vector)
        ae_vector = jnp.where(ae_vector > 0.25 * unit_cell_length, unit_cell_length**2/(8.*(unit_cell_length - 2.*ae_vector)), ae_vector)
        ''' this line can be used to enforce the boundary condition at 1/2 if necessary as a test. However, if the minimum image convention
        holds then it is never applied. '''
        # ae_vector = jnp.where(jnp.abs(ae_vector) > 0.5 * unit_cell_length, jnp.inf, ae_vector)
    
    # print('ae_vector')
    # print(ae_vector)
    pre_activation = jnp.matmul(ae_vector, sigma) + d0
    x = jnp.isnan(pre_activation)
    print(x.any())
    pre_activation = jnp.where(x, jnp.ones_like(pre_activation)*jnp.inf, pre_activation)
    pre_activation = jnp.where(jnp.isinf(pre_activation), 99999999999999., pre_activation)
    # print('pre', sigma.shape)
    # break_if_nan(pre_activation)
    exponent = pre_activation.reshape(n_spin, 3, -1, n_spin, 1, order='F')
    # exponent = jnp.linalg.norm(exponent, axis=1)
    exponent = norm(exponent, axis=1)
    # print('exp')
    # break_if_nan(exponent)
    exponential = jnp.exp(-exponent)
    # print(exponential)
    return ae_vector, exponential

def env_sigma_i(sigmas: jnp.array,
                ae_vectors: jnp.array,
                activations: list,
                d0s: jnp.array,
                _compute_exponents: anisotropic_exponent) -> jnp.array:

    # sigma (n_det, n_spin_i, n_atom, 3, 3)
    # ae_vectors (n_spin_j, n_atom, 3)

    n_spin, n_atom, _ = ae_vectors.shape
    ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    outs = []
    for ae_vector, sigma, d0 in zip(ae_vectors, sigmas, d0s):

        activation, exponential = _compute_exponents(ae_vector, sigma, d0, n_spin)

        activations.append(activation)
        
        outs.append(exponential)

    return jnp.concatenate(outs, axis=-1)


def create_compute_orbital_exponents(orbitals='anisotropic', 
                                     periodic_boundaries=False,
                                     unit_cell_length=1.):

    if orbitals == 'anisotropic':
        _compute_exponent = partial(anisotropic_exponent, periodic_boundaries=periodic_boundaries, unit_cell_length=unit_cell_length)
            
    return _compute_exponent


def apply_minimum_image_convention(displacement_vectors, unit_cell_length=1.):
    '''
    pseudocode:
        - translate to the unit cell 
        - compute the distances
        - 2 * element distances will be maximum 0.999 (as always in the same cell)
        - int(2 * element distances) will either be 0, 1 or -1
    '''
    displace = (2. * displacement_vectors / unit_cell_length).astype(int).astype(displacement_vectors.dtype) * unit_cell_length
    displacement_vectors = displacement_vectors - lax.stop_gradient(displace)  # 
    return displacement_vectors


def compute_ae_vectors_periodic_i(walkers: jnp.array, r_atoms: jnp.array, unit_cell_length: float=1.) -> jnp.array:
    ''' computes the nuclei-electron displacement vectors under the minimum image convention '''
    r_atoms = jnp.expand_dims(r_atoms, axis=0)
    walkers = jnp.expand_dims(walkers, axis=1)
    ae_vectors = r_atoms - walkers
    ae_vectors = apply_minimum_image_convention(ae_vectors, unit_cell_length)
    return ae_vectors


def env_pi_i(pis: jnp.array,
             exponential: jnp.array,
             activations: list,
             d0s) -> jnp.array:
    # exponential (j k i m)
    # factor (k i j)

    n_spins, n_det = exponential.shape[:2]

    exponential = [jnp.squeeze(x, axis=(1, 2))
                   for y in jnp.split(exponential, n_spins, axis=2)
                   for x in jnp.split(y, n_det, axis=1)]

    [activations.append(x) for x in exponential]
    # [print((e @ pi).shape, d0.shape) for pi, e, d0 in zip(pis, exponential, d0s)]
    orbitals = jnp.stack([(e @ pi) + d0 for pi, e, d0 in zip(pis, exponential, d0s)], axis=-1)
    # print(factor.shape, orbitals.shape)
    return jnp.transpose(orbitals.reshape(n_spins, n_det, n_spins), (1, 2, 0))


def logabssumdet(orb_up: jnp.array,
                 orb_down: jnp.array) -> jnp.array:
    s_up, log_up = jnp.linalg.slogdet(orb_up)
    s_down, log_down = jnp.linalg.slogdet(orb_down)

    logdet_sum = log_up + log_down
    logdet_max = jnp.max(logdet_sum)

    argument = s_up * s_down * jnp.exp(logdet_sum - logdet_max)
    sum_argument = jnp.sum(argument, axis=0)
    sign = jnp.sign(sum_argument)

    return jnp.log(jnp.abs(sum_argument)) + logdet_max, sign

exponent_fn = create_compute_orbital_exponents(periodic_boundaries=mol.periodic_boundaries, unit_cell_length=mol.unit_cell_length)
_env_sigma_i = partial(env_sigma_i, _compute_exponents=exponent_fn)


activations = []

walkers = np.array(walkers[0, 0])
walkers[0] = np.array(mol.r_atoms[1]) - mol.unit_cell_length
walkers = jnp.array(walkers)

print(walkers.shape)

compute_ae_vectors_i = partial(compute_ae_vectors_periodic_i, unit_cell_length=mol.unit_cell_length)

ae_vectors = compute_ae_vectors_i(walkers, mol.r_atoms)

ae_up, ae_down = jnp.split(ae_vectors, [mol.n_up], axis=0)

exp_up = _env_sigma_i([jnp.abs(x) for x in (params['envelopes']['sigma']['up'])], ae_up, activations, d0s['envelopes']['sigma']['up']) ##
exp_down = _env_sigma_i(params['envelopes']['sigma']['down'], ae_down, activations, d0s['envelopes']['sigma']['down']) ##

orb_up = env_pi_i(params['envelopes']['pi'][0], exp_up, activations, d0s['envelopes']['pi'][0])
orb_down = env_pi_i(params['envelopes']['pi'][1], exp_down, activations, d0s['envelopes']['pi'][1])

log_psi, sign = logabssumdet(orb_up, orb_down)


print(exp_up.shape, exp_up)
print(orb_up.shape, orb_up)
s_up, log_up = jnp.linalg.slogdet(orb_up)
print(s_up.shape, s_up, log_up.shape, log_up)
print(log_psi)

kinetic_energy = ke(params, walkers[None, None, ...])

print(kinetic_energy)

from jax import grad

wf_new = lambda walkers: vwf(params, walkers).sum()
grad_f = jax.grad(wf_new)

grads = grad_f(walkers[None, ...])


print(grads)
