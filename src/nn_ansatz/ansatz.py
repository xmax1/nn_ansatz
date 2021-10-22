from typing import Callable
from functools import partial

from itertools import product

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np

from .utils import remove_aux
from .parameters import expand_d0s, initialise_d0s
from .heg_ansatz import create_heg_wf, generate_k_points
from .ansatz_base import *


def create_wf(mol, kfac: bool=False, orbitals: bool=False, signed: bool=False):
    ''' initializes the wave function ansatz for various applications '''

    print('creating wf')
  
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph, mol.n_in)

    _compute_single_stream_vectors = partial(compute_single_stream_vectors_i, r_atoms=mol.r_atoms, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis)
    _compute_inputs = partial(compute_inputs_i, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis)
    _compute_orbitals, _sum_orbitals = create_orbitals(orbitals=mol.orbitals, n_el=mol.n_el, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis)


    _wf_orbitals = partial(wf_orbitals, 
                           masks=masks,
                           n_el=mol.n_el,
                           n_up=mol.n_up,
                           n_down=mol.n_down,
                           _compute_single_stream_vectors=_compute_single_stream_vectors,
                           _compute_inputs=_compute_inputs,
                           _compute_orbitals=_compute_orbitals,
                           _sum_orbitals=_sum_orbitals)

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


def wf_orbitals(params: dict, 
                walkers: jnp.array, 
                d0s: dict, 
                masks,
                n_up: int,
                n_down: int,
                n_el: int,
                _compute_single_stream_vectors: Callable,
                _compute_inputs_i: Callable,
                _compute_orbitals: Callable,
                _sum_orbitals: Callable):

    if len(walkers.shape) == 1:  # this is a hack to get around the jvp
        walkers = walkers.reshape(n_el, 3)

    activations = []

    single_stream_vectors = _compute_single_stream_vectors(walkers) ## 

    single, pairwise = _compute_inputs_i(walkers, single_stream_vectors) ##

    single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *masks[0])

    split = linear_split(params['split0'], split, activations, d0s['split0'])
    single = linear(params['s0'], single_mixed, split, activations, d0s['s0'])
    pairwise = linear_pairwise(params['p0'], pairwise, activations, d0s['p0'])

    for i, mask in enumerate(masks[1:], 1):
        single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *mask)

        split = linear_split(params['split%i'%i], split, activations, d0s['split%i'%i])
        single = linear(params['s%i'%i], single_mixed, split, activations, d0s['s%i'%i]) + single
        pairwise = linear_pairwise(params['p%i'%i], pairwise, activations, d0s['p%i'%i]) + pairwise

    data_up, data_down = jnp.split(single, [n_up], axis=0)

    factor_up = env_linear_i(params['env_lin_up'], data_up, activations, d0s['env_lin_up'])
    if not n_up == n_el: factor_down = env_linear_i(params['env_lin_down'], data_down, activations, d0s['env_lin_down'])

    single_stream_vectors = split_and_squeeze(single_stream_vectors, axis=1)

    exp_up = []
    exp_down = []
    for m, single_stream_vector in enumerate(single_stream_vectors):
        ss_up_m, ss_down_m = jnp.split(single_stream_vector, [n_up], axis=0)
        exp_up_m = _compute_orbitals(params.get('env_sigma_up_m%i' % m), ss_up_m, activations, d0s.get('env_sigma_up_m%i' % m)) ##
        exp_up.append(exp_up_m)
        
        if not n_up == n_el:
            exp_down_m = _compute_orbitals(params.get('env_sigma_down_m%i' % m), ss_down_m, activations, d0s.get('env_sigma_down_m%i' % m)) ##
            exp_down.append(exp_down_m)
    
    exp_up = jnp.stack(exp_up, axis=-1)
    orb_up = _sum_orbitals(params, factor_up, exp_up, activations, 'up', d0s)
    
    orb_down = None
    if not n_up == n_el:
        exp_down = jnp.stack(exp_down, axis=-1)
        orb_down = _sum_orbitals(params, factor_down, exp_down, activations, 'down', d0s)

    return orb_up, orb_down, activations


def create_orbitals(orbitals='anisotropic',
                    n_el: Optional[int]=None,
                    basis: Optional[jnp.array]=None,
                    inv_basis: Optional[jnp.array]=None,
                    pbc: bool=False):

    if orbitals == 'anisotropic':
        _compute_orbitals = partial(anisotropic_orbitals, basis=basis, inv_basis=inv_basis, pbc=pbc)
        _sum_orbitals = env_pi_i

    if orbitals == 'isotropic':
        _compute_orbitals = isotropic_orbitals()
        _sum_orbitals = env_pi_i
    
    if orbitals == 'real_plane_waves':
        shells = [1, 7, 19]
        shell = shells.index(n_el) + 1
        k_points = generate_k_points(n_shells=shell) * 2*jnp.pi
        k_points = transform_vector_space(k_points, inv_basis)
        _compute_orbitals = partial(real_plane_wave_orbitals, k_points=k_points)
        def _sum_orbitals(params: jnp.array,
                            factor: jnp.array,
                            orbital: jnp.array,
                            activations: list,
                            spin: str,
                            d0s):
            return factor * orbital

    return _compute_orbitals, _sum_orbitals


def anisotropic_orbitals(sigma,
                         orb_vector,
                         d0,
                         activations: Optional[list]=None,
                         basis: Optional[jnp.array]=None,
                         inv_basis: Optional[jnp.array]=None,
                         pbc: bool=False,
                         eps=0.0001):
    # sigma (3, 3 * n_det * n_spin)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    n_spin = orb_vector.shape[0]
    
    if pbc:
        # ae_vector = jnp.where(jnp.abs(ae_vector) == 0.5 * unit_cell_length, ae_vector + 0.000001, ae_vector)
        orb_vector = transform_vector_space(orb_vector, inv_basis)
        orb_vector = jnp.where(orb_vector <= -0.25, -1.**2/(8.*(1. + 2.*orb_vector)+eps), orb_vector)
        orb_vector = jnp.where(orb_vector >= 0.25 * 1., 1.**2/(8.*(1. - 2.*orb_vector)-eps), orb_vector)
        orb_vector = transform_vector_space(orb_vector, basis)
        ''' this line can be used to enforce the boundary condition at 1/2 if necessary as a test. However, if the minimum image convention
        holds then it is never applied. '''
        # ae_vector = jnp.where(jnp.abs(ae_vector) == 0.5 * unit_cell_length, jnp.inf, ae_vector_tmp)
        # ae_vector = jnp.where(jnp.isinf(ae_vector), jnp.inf, ae_vector)
    
    pre_activation = jnp.matmul(orb_vector, sigma)  + d0  # n_spin_j, 3 * n_det * n_spin
    '''this line is required to force the situations where -jnp.inf + jnp.inf = jnp.nan creates a nan from the exponential and not a zero 
    (else jnp.inf + jnp.inf = jnp.inf)'''
    # pre_activation = jnp.where(jnp.isnan(pre_activation), jnp.inf, pre_activation)

    # the way the activations are unpacked is important, order check in /home/amawi/projects/nn_ansatz/src/scripts/debugging/shapes/sigma_reshape.py
    # surprisingly the alternate way 'works' but there is a difference in performance which is notable at larger system sizes
    exponent = pre_activation.reshape(n_spin, 3, -1, order='F')  # order ='F'
    exponent = jnp.linalg.norm(exponent, axis=1)
    exponential = jnp.exp(-exponent)

    if not activations is None: activations.append(orb_vector)
    return exponential


def isotropic_orbitals(sigma, 
                       orb_vector,                      
                       d0, 
                       activations: Optional[list]=None,
                       basis: Optional[jnp.array]=None,
                       inv_basis: Optional[jnp.array]=None,
                       pbc: bool=False,):
    # sigma (n_det, n_spin_i)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    n_spin = orb_vector.shape[0]
    
    
    
    if not pbc:
        norm = jnp.linalg.norm(orb_vector, axis=-1)[..., None] # (n_spin,)
        exponent = (norm  * sigma + d0).reshape(n_spin, -1, n_spin, 1, order='F')
        exponential = jnp.exp(-exponent)
    else:
        orb_vector = transform_vector_space(orb_vector, inv_basis)
        norm = jnp.linalg.norm(orb_vector, axis=-1)[..., None] # (n_spin,)
        exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(unit_cell_length - norm) * sigma + d0) - 2 * jnp.exp(-sigma * unit_cell_length / 2.)
        exponential = jnp.where(norm > unit_cell_length/2., 0.0, exponential)
        exponential = exponential.reshape(n_spin, -1, n_spin, 1, order='F')

        # exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(1. - norm) * sigma + d0) - 2 * jnp.exp(-(1. / 2.) * sigma)
        # exponential = jnp.where(norm < ucl2, exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

        # tr_ae_vector = ae_vector.dot(inv_real_basis)
        # tr_norm = jnp.linalg.norm(tr_ae_vector, axis=-1)
        # exponential = jnp.where(tr_norm < min_cell_width / 2., exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

    return norm, exponential


def generate_k_points(n_shells=3):
    img_range = jnp.arange(-3, 3+1)  # preset, when are we ever going to use more
    img_sets = jnp.array(list(product(*[img_range, img_range, img_range])))
    norms = jnp.linalg.norm(img_sets, axis=-1)
    idxs = jnp.argsort(norms)
    img_sets, norms = img_sets[idxs], norms[idxs]
    norm = 0.
    k_shells = {norm: [jnp.array([0.0, 0.0, 0.0])]}  # leacing the dictionary logic in case we ever need this data structure
    for k_point, norm_tmp in zip(img_sets[1:], norms[1:]):
        if norm_tmp > norm:
            if len(k_shells) == n_shells:
                break
            norm = norm_tmp
            k_shells[norm] = [k_point]
        else:
            if np.any([(k_point == x).all() for x in k_shells[norm]]):
                continue # because we include the opposite k_point in the sequence this statement avoids repeats
            k_shells[norm].append(k_point)
        k_shells[norm].append(-k_point)
    k_points = []
    for k, v in k_shells.items():
        for k_point in v:
            k_points.append(k_point)
    return jnp.array(k_points)


def real_plane_wave_orbitals(sigma,
                             orb_vector,
                             d0,
                             activations: Optional[list]=None,
                             k_points=jnp.array) -> jnp.array:

    # sigma (n_det, n_spin_i, n_atom, 3, 3)
    # walkers (n_spin_j, 3)

    n_el = orb_vector.shape[0]
    args = orb_vector @ k_points.T  # (n_el, n_el)
    args = jnp.split(args, n_el, axis=1)
    pf = [jnp.cos, jnp.sin]
    dets = []
    for i, arg in enumerate(args):
        column = pf[i%2](arg)
        dets.append(column)
    dets = jnp.concatenate(dets, axis=-1)
    return dets


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
    s_down, log_down = jnp.linalg.slogdet(orb_down) if not orb_down is None else jnp.ones_like(s_up), jnp.zeros_like(log_up)

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

    # --- Pairwise summations
    sum_pairwise = jnp.repeat(jnp.expand_dims(pairwise, axis=0), n_el, axis=0)

    # up
    sum_pairwise_up = pairwise_up_mask * sum_pairwise
    sum_pairwise_up = jnp.sum(sum_pairwise_up, axis=1) / float(n_up)

    # down
    if n_down > 0:
        sum_spin_down = single_down_mask * single
        sum_spin_down = jnp.sum(sum_spin_down, axis=0, keepdims=True) / float(n_down)

        # down
        sum_pairwise_down = pairwise_down_mask * sum_pairwise
        sum_pairwise_down = jnp.sum(sum_pairwise_down, axis=1) / float(n_down)
    #     sum_spin_down = jnp.repeat(sum_spin_down, n_el, axis=1) # not needed in split

    
        single = jnp.concatenate((single, sum_pairwise_up, sum_pairwise_down), axis=1)
        split = jnp.concatenate((sum_spin_up, sum_spin_down), axis=1)
        return single, split
    else:
        single = jnp.concatenate((single, sum_pairwise_up), axis=1)
        split = sum_spin_up
        return single, split

