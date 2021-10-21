from typing import Callable
from functools import partial
from itertools import product

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np
import typing

from .utils import remove_aux
from .parameters import expand_d0s, initialise_d0s
from .ansatz_base import *

### NOTES
# parameters problem: don't generate sigmas - delete for now and maybe fix later if results good


def create_heg_wf(mol, kfac: bool=False, orbitals: bool=False, signed: bool=False):
    ''' initializes the wave function ansatz for various applications '''

    print('creating wf')
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph, mol.n_in)
    
    shell = [1, 7, 19].index(mol.n_el) + 1
    k_points = generate_k_points(n_shells=shell) * 2*jnp.pi / mol.unit_cell_length
    _env_sigma_i = partial(env_sigma_i, k_points=k_points)

    _compute_e_vectors_i = partial(compute_e_vectors_periodic_i, unit_cell_length=mol.unit_cell_length)

    _compute_inputs_i = create_compute_inputs_i(mol)

    _wf_orbitals = partial(wf_orbitals, 
                           mol=mol, 
                           masks=masks,
                           spin_polarized=mol.spin_polarized,
                           _compute_inputs_i=_compute_inputs_i, 
                           _env_sigma_i=_env_sigma_i,
                           _compute_e_vectors_i=_compute_e_vectors_i)

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
                spin_polarized: bool,
                _env_sigma_i: Callable,
                _compute_inputs_i: Callable,
                **kwargs):


    if len(walkers.shape) == 1:  # this is a hack to get around the jvp
        walkers = walkers.reshape(mol.n_up + mol.n_down, 3)

    activations = []

    single, pairwise = _compute_inputs_i(walkers) ##

    single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *masks[0])

    split = linear_split(params['split0'], split, activations, d0s['split0'])
    single = linear(params['s0'], single_mixed, split, activations, d0s['s0'])
    pairwise = linear_pairwise(params['p0'], pairwise, activations, d0s['p0'])

    for i, mask in enumerate(masks[1:], 1):
        single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *mask)

        split = linear_split(params['split%i'%i], split, activations, d0s['split%i'%i])
        single = linear(params['s%i'%i], single_mixed, split, activations, d0s['s%i'%i]) + single
        pairwise = linear_pairwise(params['p%i'%i], pairwise, activations, d0s['p%i'%i]) + pairwise

    walkers_up, walkers_down = jnp.split(walkers, [mol.n_up], axis=0)
    data_up, data_down = jnp.split(single, [mol.n_up], axis=0)

    factor_up = env_linear_i(params, data_up, activations, d0s, 'env_lin_up')
    if not spin_polarized: factor_down = env_linear_i(params, data_down, activations, d0s, 'env_lin_down')

    exp_up = _env_sigma_i(walkers_up)
    if not spin_polarized: exp_down = _env_sigma_i(walkers_down) 

    orb_up = factor_up * exp_up
    orb_down = factor_down * exp_down if not spin_polarized else None
    
    return orb_up, orb_down, activations


def compute_e_vectors_periodic_i(walkers: jnp.array, unit_cell_length: float) -> jnp.array:
    return apply_minimum_image_convention(jnp.expand_dims(walkers, axis=1), unit_cell_length)


def create_compute_inputs_i(mol):

    _compute_inputs_periodic_i = partial(compute_inputs_periodic_i,
                                            n_periodic_input=mol.n_periodic_input, 
                                            unit_cell_length=mol.unit_cell_length)
    
    return _compute_inputs_periodic_i


def compute_inputs_periodic_i(walkers: jnp.array, n_periodic_input: int, unit_cell_length: float=1.):

    e_distances = jnp.linalg.norm(walkers, axis=-1, keepdims=True)
    e_vectors = walkers - (jnp.ones_like(walkers) * unit_cell_length / 2.)  # enforces symmetry with the edge of the box
    e_vectors_periodic = input_activation(e_vectors, unit_cell_length, n_periodic_input)
    single_inputs = jnp.concatenate([e_vectors_periodic, e_distances], axis=-1)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_vectors = apply_minimum_image_convention(ee_vectors, unit_cell_length)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    ee_vectors_periodic = input_activation(ee_vectors, unit_cell_length, n_periodic_input)
    pairwise_inputs = jnp.concatenate([ee_vectors_periodic, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def apply_minimum_image_convention(displacement_vectors, unit_cell_length=1.):
    '''
    pseudocode:
        - translate to the unit cell 
        - compute the distances
        - 2 * element distances will be maximum 0.999 (as always in the same cell)
        - int(2 * element distances) will either be 0, 1 or -1
    '''
    displace = (2. * displacement_vectors / unit_cell_length).astype(int).astype(displacement_vectors.dtype) * unit_cell_length
    # displacement_vectors = displacement_vectors - lax.stop_gradient(displace)  #
    displacement_vectors = displacement_vectors - displace #
    return displacement_vectors



def env_linear_i(params: jnp.array,
                 data: jnp.array,
                 activations: list,
                 d0s: jnp.array,
                 key: str) -> jnp.array:
    '''
    bias = jnp.ones((data.shape[0], 1))
    data = jnp.concatenate((data, bias), axis=1)
    out = jnp.einsum('jf,kif->kij', data, params)
    print(out.shape)
    print(params.shape, pre_activations.shape, activation.shape)
    '''
    # params (f, k * i)
    # data (j, f)
    n_spins = data.shape[0]

    bias = jnp.ones((n_spins, 1))
    activation = jnp.concatenate((data, bias), axis=1)
    activations.append(activation)
    pre_activations = jnp.matmul(activation, params[key]) + d0s[key]  # (j, k * i)
    pre_activations = jnp.transpose(pre_activations).reshape(-1, n_spins, n_spins)  # (k, i, j)
    return pre_activations




def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(tensor.shape[axis], axis=axis)]




# create the set of k-points, surely there is a better way smh

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


def env_sigma_i(walkers: jnp.array,
                k_points: jnp.array) -> jnp.array:

    # sigma (n_det, n_spin_i, n_atom, 3, 3)
    # walkers (n_spin_j, 3)

    n_el = walkers.shape[0]
    args = walkers @ k_points.T  # (n_el, n_el)
    args = jnp.split(args, n_el, axis=1)
    pf = [jnp.cos, jnp.sin]
    dets = []
    for i, arg in enumerate(args):
        column = pf[i%2](arg)
        dets.append(column)
    dets = jnp.concatenate(dets, axis=-1)
        
    return dets


def logabssumdet(orb_up: jnp.array,
                 orb_down: typing.Optional[jnp.array]=None) -> jnp.array:
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

    single = jnp.concatenate((single, sum_pairwise_up), axis=1)
    split = sum_spin_up
    return single, split
    

