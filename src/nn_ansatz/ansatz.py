from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np

from .utils import remove_aux
from .parameters import expand_d0s, initialise_d0s
from .heg_ansatz import create_heg_wf
from .ansatz_base import *


def create_wf(mol, kfac: bool=False, orbitals: bool=False, signed: bool=False):
    ''' initializes the wave function ansatz for various applications '''

    print('creating wf')
    if mol.system == 'HEG': return create_heg_wf(mol, kfac=kfac, orbitals=orbitals, signed=signed)
    
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph, mol.n_in)

    _compute_exponents = create_compute_orbital_exponents(periodic_boundaries=mol.periodic_boundaries,
                                                          orbitals=mol.orbitals,
                                                          unit_cell_length=mol.unit_cell_length)
    
    _env_sigma_i = partial(env_sigma_i, _compute_exponents=_compute_exponents)

    _compute_ae_vectors_i = compute_ae_vectors_i
    if mol.periodic_boundaries:
        _compute_ae_vectors_i = partial(compute_ae_vectors_periodic_i, unit_cell_length=mol.unit_cell_length)

    _compute_inputs_i = create_compute_inputs_i(mol)

    _wf_orbitals = partial(wf_orbitals, 
                           mol=mol, 
                           masks=masks, 
                           inv_real_basis=mol.inv_real_basis,
                           periodic_boundaries=mol.periodic_boundaries,
                           _compute_inputs_i=_compute_inputs_i, 
                           _env_sigma_i=_env_sigma_i,
                           _compute_ae_vectors_i=_compute_ae_vectors_i)

    def _signed_wf(params, walkers, d0s):
        orb_up, orb_down, _ = _wf_orbitals(params, walkers, d0s)
        log_psi, sign = logabssumdet(orb_up, orb_down)
        return log_psi, sign

    def _wf(params, walkers, d0s):
        orb_up, orb_down, _ = _wf_orbitals(params, walkers, d0s)
        log_psi, _ = logabssumdet(orb_up, orb_down)
        return log_psi

    def _kfac_wf(params, walkers, d0s):
        orb_up, orb_down, activations = _wf_orbitals(params, walkers, d0s)
        log_psi, _ = logabssumdet(orb_up, orb_down)
        return log_psi, activations
    
    d0s = initialise_d0s(mol)

    if signed:
        _partial_wf = partial(_signed_wf, d0s=d0s)
        _vwf = vmap(_partial_wf, in_axes=(None, 0))
        return _vwf

    if orbitals:
        _wf_orbitals_remove_activations = remove_aux(_wf_orbitals, axis=1)
        return partial(_wf_orbitals_remove_activations, d0s=d0s)

    if kfac:
        return vmap(_kfac_wf, in_axes=(None, 0, 0))

    _partial_wf = partial(_wf, d0s=d0s)
    _vwf = vmap(_partial_wf, in_axes=(None, 0))
    
    return _vwf


def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(1, axis=axis)]


def wf_orbitals(params, 
                walkers, 
                d0s, 
                mol, 
                masks,
                periodic_boundaries: bool,
                inv_real_basis: jnp.array,
                _compute_ae_vectors_i: Callable,
                _compute_inputs_i: Callable, 
                _env_sigma_i: Callable):


    if len(walkers.shape) == 1:  # this is a hack to get around the jvp
        walkers = walkers.reshape(mol.n_up + mol.n_down, 3)

    activations = []

    ae_vectors = _compute_ae_vectors_i(walkers, mol.r_atoms) ## 

    single, pairwise = _compute_inputs_i(walkers, ae_vectors) ##

    single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *masks[0])

    split = linear_split(params['split0'], split, activations, d0s['split0'])
    single = linear(params['s0'], single_mixed, split, activations, d0s['s0'])
    pairwise = linear_pairwise(params['p0'], pairwise, activations, d0s['p0'])

    for i, mask in enumerate(masks[1:], 1):
        single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *mask)

        split = linear_split(params['split%i'%i], split, activations, d0s['split%i'%i])
        single = linear(params['s%i'%i], single_mixed, split, activations, d0s['s%i'%i]) + single
        pairwise = linear_pairwise(params['p%i'%i], pairwise, activations, d0s['p%i'%i]) + pairwise

    ae_up, ae_down = jnp.split(ae_vectors, [mol.n_up], axis=0)
    data_up, data_down = jnp.split(single, [mol.n_up], axis=0)

    factor_up = env_linear_i(params['env_lin_up'], data_up, activations, d0s['env_lin_up'])
    factor_down = env_linear_i(params['env_lin_down'], data_down, activations, d0s['env_lin_down'])

    ae_up = split_and_squeeze(ae_up, axis=1)
    ae_down = split_and_squeeze(ae_down, axis=1)

    exp_up = []
    exp_down = []
    for m, (ae_up_m, ae_down_m) in enumerate(zip(ae_up, ae_down)):
        exp_up_m = _env_sigma_i(params['env_sigma_up_m%i' % m], ae_up_m, activations, d0s['env_sigma_up_m%i' % m]) ##
        exp_down_m = _env_sigma_i(params['env_sigma_down_m%i' % m], ae_down_m, activations, d0s['env_sigma_down_m%i' % m]) ##
        exp_up.append(exp_up_m)
        exp_down.append(exp_down_m)
    exp_up = jnp.stack(exp_up, axis=-1)
    exp_down = jnp.stack(exp_down, axis=-1)

    orb_up = env_pi_i(params, factor_up, exp_up, activations, 'up', d0s)
    orb_down = env_pi_i(params, factor_down, exp_down, activations, 'down', d0s)
    return orb_up, orb_down, activations



def anisotropic_exponent(sigma,
                         ae_vector,
                         d0, 
                         n_spin, 
                         unit_cell_length=1., 
                         periodic_boundaries=False, 
                         eps=0.0001):
    # sigma (3, 3 * n_det * n_spin)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    
    if periodic_boundaries:
        # ae_vector = jnp.where(jnp.abs(ae_vector) == 0.5 * unit_cell_length, ae_vector + 0.000001, ae_vector)
        ae_vector = jnp.where(ae_vector <= -0.25 * unit_cell_length, -unit_cell_length**2/(8.*(unit_cell_length + 2.*ae_vector)+eps), ae_vector)
        ae_vector = jnp.where(ae_vector >= 0.25 * unit_cell_length, unit_cell_length**2/(8.*(unit_cell_length - 2.*ae_vector)-eps), ae_vector)
        ''' this line can be used to enforce the boundary condition at 1/2 if necessary as a test. However, if the minimum image convention
        holds then it is never applied. '''
        # ae_vector = jnp.where(jnp.abs(ae_vector) == 0.5 * unit_cell_length, jnp.inf, ae_vector_tmp)
        # ae_vector = jnp.where(jnp.isinf(ae_vector), jnp.inf, ae_vector)
    
    pre_activation = jnp.matmul(ae_vector, sigma)  + d0  # n_spin_j, 3 * n_det * n_spin
    '''this line is required to force the situations where -jnp.inf + jnp.inf = jnp.nan creates a nan from the exponential and not a zero 
    (else jnp.inf + jnp.inf = jnp.inf)'''
    # pre_activation = jnp.where(jnp.isnan(pre_activation), jnp.inf, pre_activation)

    # the way the activations are unpacked is important, order check in /home/amawi/projects/nn_ansatz/src/scripts/debugging/shapes/sigma_reshape.py
    # surprisingly the alternate way 'works' but there is a difference in performance which is notable at larger system sizes
    exponent = pre_activation.reshape(n_spin, 3, -1, order='F')  # order ='F'
    exponent = jnp.linalg.norm(exponent, axis=1)
    exponential = jnp.exp(-exponent)
    return ae_vector, exponential


def isotropic_exponent(ae_vector, sigma, d0, n_spin, 
                       periodic_boundaries=False,
                       unit_cell_length=1.):
    # sigma (n_det, n_spin_i)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    
    norm = jnp.linalg.norm(ae_vector, axis=-1)[..., None] # (n_spin,)
    
    if not periodic_boundaries:
        exponent = (norm  * sigma + d0).reshape(n_spin, -1, n_spin, 1, order='F')
        exponential = jnp.exp(-exponent)
    else:
        exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(unit_cell_length - norm) * sigma + d0) - 2 * jnp.exp(-sigma * unit_cell_length / 2.)
        exponential = jnp.where(norm > unit_cell_length/2., 0.0, exponential)
        exponential = exponential.reshape(n_spin, -1, n_spin, 1, order='F')

        # exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(1. - norm) * sigma + d0) - 2 * jnp.exp(-(1. / 2.) * sigma)
        # exponential = jnp.where(norm < ucl2, exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

        # tr_ae_vector = ae_vector.dot(inv_real_basis)
        # tr_norm = jnp.linalg.norm(tr_ae_vector, axis=-1)
        # exponential = jnp.where(tr_norm < min_cell_width / 2., exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

    return norm, exponential


def env_sigma_i(sigma: jnp.array,
                ae_vector: jnp.array,
                activations: list,
                d0: jnp.array,
                _compute_exponents: Callable=anisotropic_exponent) -> jnp.array:
    """

    Notes:
        This is very strange. This implementation allows us to create the matrices for KFAC
        However, using this over the einsum version causes the energies of the pretraining samples and samples direct
        from the wave function to diverge

        I have adjusted the initialisation to account for this, but still diverges.
        It is the better implementation and the effect disappears when the VMC starts
        VERY WEIRD

        It is dependent on how the sigmas are constructed
        reshape(n_spin, -1, n_spin, 3) and
        .reshape(n_spin, 3, -1, n_spin, order='F') when constructed
        p = [jnp.concatenate([init_linear_layer(k, new_shape, bias)[..., None].reshape(3, -1, order='F') for k in subkeys], axis=-1)
             for _ in range(n_atom)]

        really need to be careful of this as the initialisation will change
        luckily we'll probably be dropping the anisotropic decay soon but will still need this for comparison

        this makes the most sense
        construct
        p = [jnp.concatenate([init_linear_layer(k, new_shape, bias) for k in subkeys], axis=-1)
             for _ in range(n_atom)]

        then unroll Fortran style 'from the left' when doing the matmul putting the 3 at the start
        
        exponent = jnp.einsum('jmv,kimvc->jkimc', ae_vectors, sigma)
        return jnp.exp(-jnp.linalg.norm(exponent, axis=-1))
    """
    # sigma (n_det, n_spin_i, n_atom, 3, 3)
    # ae_vectors (n_spin_j, n_atom, 3)

    n_spin = ae_vector.shape[0]
    # ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    # ae_vectors = split_and_squeeze(ae_vectors, 1)
    # d0s = split_and_squeeze(d0s, 0)
    # sigmas = split_and_squeeze(sigmas, 0)
    # for ae_vector, sigma, d0 in zip(ae_vectors, sigmas, d0s):

    activation, exponential = _compute_exponents(sigma, ae_vector, d0, n_spin)

    activations.append(activation)
        
    return exponential


def create_compute_orbital_exponents(orbitals='anisotropic', 
                                     periodic_boundaries=False,
                                     unit_cell_length=1.):

    if orbitals == 'anisotropic':
        _compute_exponent = partial(anisotropic_exponent, periodic_boundaries=periodic_boundaries, unit_cell_length=unit_cell_length)
            
    elif orbitals == 'isotropic':
        _compute_exponent = partial(isotropic_exponent, periodic_boundaries=periodic_boundaries, unit_cell_length=unit_cell_length)
        
    return _compute_exponent




def env_pi_i(params: jnp.array,
             factor: jnp.array,
             exponential: jnp.array,
             activations: list,
             spin: str,
             d0s) -> jnp.array:
    # exponential (j k*i m)
    # factor (k i j)

    # Einsum sanity check
    # orbitals = factor * jnp.einsum('jkim,kim->kij', exponential, pi)
    n_det, n_spins = factor.shape[:2]

    # n_det * n_spin of (n_spin, n_atom)
    exponential = [jnp.squeeze(x, axis=1) for x in jnp.split(exponential, n_spins*n_det, axis=1)]  

    [activations.append(x) for x in exponential]

    orbitals = jnp.stack([(e @ params['env_pi_%s_%i' % (spin, i)]) + d0s['env_pi_%s_%i' % (spin, i)] 
                           for i, e in enumerate(exponential)], axis=-1)

    return factor * jnp.transpose(orbitals.reshape(n_spins, n_det, n_spins), (1, 2, 0))


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


def mixer_i(single: jnp.array,
            pairwise: jnp.array,
            n_el,
            n_up,
            n_down,
            single_up_mask,
            single_down_mask,
            pairwise_up_mask,
            pairwise_down_mask):
    # single (n_samples, n_el, n_single_features)
    # pairwise (n_samples, n_pairwise, n_pairwise_features)

    # --- Single summations
    # up
    sum_spin_up = single_up_mask * single
    sum_spin_up = jnp.sum(sum_spin_up, axis=0, keepdims=True) / float(n_up)
    #     sum_spin_up = jnp.repeat(sum_spin_up, n_el, axis=1)  # not needed in split

    # down
    sum_spin_down = single_down_mask * single
    sum_spin_down = jnp.sum(sum_spin_down, axis=0, keepdims=True) / float(n_down)
    #     sum_spin_down = jnp.repeat(sum_spin_down, n_el, axis=1) # not needed in split

    # --- Pairwise summations
    sum_pairwise = jnp.repeat(jnp.expand_dims(pairwise, axis=0), n_el, axis=0)

    # up
    sum_pairwise_up = pairwise_up_mask * sum_pairwise
    sum_pairwise_up = jnp.sum(sum_pairwise_up, axis=1) / float(n_up)

    # down
    sum_pairwise_down = pairwise_down_mask * sum_pairwise
    sum_pairwise_down = jnp.sum(sum_pairwise_down, axis=1) / float(n_down)

    single = jnp.concatenate((single, sum_pairwise_up, sum_pairwise_down), axis=1)
    split = jnp.concatenate((sum_spin_up, sum_spin_down), axis=1)
    return single, split


