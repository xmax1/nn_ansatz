
from typing import Callable, Optional
from functools import partial
from .sampling import transform_vector_space

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np


def compute_single_stream_vectors_i(walkers: jnp.array, 
                                    r_atoms: Optional[jnp.array]=None,
                                    basis: Optional[jnp.array]=None,
                                    inv_basis: Optional[jnp.array]=None,
                                    pbc: bool=False) -> jnp.array:
    ''' computes the nuclei-electron displacement vectors '''
    single_stream_vectors = jnp.expand_dims(walkers, axis=1)
    if not r_atoms is None:
        r_atoms = jnp.expand_dims(r_atoms, axis=0)
        single_stream_vectors = r_atoms - single_stream_vectors
        if pbc: 
            single_stream_vectors = apply_minimum_image_convention(single_stream_vectors, basis, inv_basis)
    return single_stream_vectors


def compute_ee_vectors_i(walkers):
    ''' computes the electron-electron displacement vectors '''
    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(re1, [1, 0, 2])
    ee_vectors = re2 - re1
    return ee_vectors



def input_activation(inputs: jnp.array, inv_basis: jnp.array):
    inputs = transform_vector_space(inputs, inv_basis)
    # return jnp.concatenate([jnp.sin((2.*i*jnp.pi / unit_cell_length) * inputs) for i in range(1, n_periodic_input+1)], axis=-1)
    return inputs**2 / jnp.exp(4.*jnp.abs(inputs))


def apply_minimum_image_convention(displacement_vectors, basis, inv_basis):
    '''
    pseudocode:
        - translate to the unit cell 
        - compute the distances
        - 2 * element distances will be maximum 0.999 (as always in the same cell)
        - int(2 * element distances) will either be 0, 1 or -1
        # displacement_vectors = displacement_vectors - lax.stop_gradient(displace)  #
    '''
    displace = (2. * transform_vector_space(displacement_vectors, inv_basis)).astype(int).astype(displacement_vectors.dtype)
    displace = transform_vector_space(displace, basis)
    displacement_vectors = displacement_vectors - displace 
    return displacement_vectors


def compute_inputs_i(walkers: jnp.array, 
                     single_stream_vectors: jnp.array, 
                     basis: Optional[jnp.array]=None,
                     inv_basis: Optional[jnp.array]=None,
                     pbc: bool=False):

    n_el = walkers.shape[0]
    n_el, n_features = single_stream_vectors.shape[:2]

    if pbc: single_stream_vectors = input_activation(single_stream_vectors, inv_basis)
    single_distances = jnp.linalg.norm(single_stream_vectors, axis=-1, keepdims=True)
    single_inputs = jnp.concatenate([single_stream_vectors, single_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_el, 4 * n_features)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    if pbc: ee_vectors = apply_minimum_image_convention(ee_vectors, basis, inv_basis)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def compute_inputs_periodic_i(walkers, ae_vectors_min_im, n_periodic_input, unit_cell_length=1.):

    n_electrons, n_atoms = ae_vectors_min_im.shape[:2]

    ae_distances = jnp.linalg.norm(ae_vectors_min_im, axis=-1, keepdims=True)
    ae_vectors_periodic = input_activation(ae_vectors_min_im, unit_cell_length, n_periodic_input)
    single_inputs = jnp.concatenate([ae_vectors_periodic, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_electrons, ((n_periodic_input * 3) + 1) * n_atoms)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_vectors = apply_minimum_image_convention(ee_vectors, unit_cell_length)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    ee_vectors_periodic = input_activation(ee_vectors, unit_cell_length, n_periodic_input)
    pairwise_inputs = jnp.concatenate([ee_vectors_periodic, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def compute_inputs_scalar_inputs_i(walkers, ae_vectors_min_im):
    ''' computes the inputs as only the distances between particles '''

    n_electrons, n_atoms = ae_vectors_min_im.shape[:2]
    single_inputs = jnp.linalg.norm(ae_vectors_min_im, axis=-1, keepdims=True)
    single_inputs = single_inputs.reshape(n_electrons, 1 * n_atoms)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_vectors_min_im = apply_minimum_image_convention(ee_vectors)
    pairwise_inputs = jnp.linalg.norm(ee_vectors_min_im, axis=-1, keepdims=True)
    return single_inputs, pairwise_inputs




def create_masks(n_atom, n_electrons, n_up, n_layers, n_sh, n_ph, n_in):

    n_sh_in, n_ph_in = n_in * n_atom, n_in

    masks = [create_masks_layer(n_sh_in, n_ph_in, n_electrons, n_up)]

    for i in range(n_layers):
        masks.append(create_masks_layer(n_sh, n_ph, n_electrons, n_up))

    return masks


def create_masks_layer(n_sh, n_ph, n_electrons, n_up):
    # single spin masks
    eye_mask = ~np.eye(n_electrons, dtype=bool)
    n_down = n_electrons - n_up
    n_pairwise = n_electrons ** 2 - n_electrons

    tmp1 = jnp.ones((n_up, n_sh))
    tmp2 = jnp.zeros((n_down, n_sh))
    single_up_mask = jnp.concatenate((tmp1, tmp2), axis=0)
    single_down_mask = (jnp.concatenate((tmp1, tmp2), axis=0) - 1.) * -1.

    # pairwise spin masks
    ups = np.ones(n_electrons)
    ups[n_up:] = 0
    downs = (ups - 1.) * -1.

    pairwise_up_mask = []
    pairwise_down_mask = []
    mask = np.zeros((n_electrons, n_electrons))

    for electron in range(n_electrons):
        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask].reshape(-1)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask].reshape(-1)

        pairwise_up_mask.append(mask_up)
        pairwise_down_mask.append(mask_down)

    pairwise_up_mask = jnp.array(pairwise_up_mask).reshape((n_electrons, n_pairwise, 1))
    pairwise_up_mask = jnp.repeat(pairwise_up_mask, n_ph, axis=-1)

    pairwise_down_mask = jnp.array(pairwise_down_mask).reshape((n_electrons, n_pairwise, 1))
    pairwise_down_mask = jnp.repeat(pairwise_down_mask, n_ph, axis=-1)
    return single_up_mask, single_down_mask, pairwise_up_mask, pairwise_down_mask
        

def drop_diagonal_i(square):
    """
    Notes:
        Previous masking code for dropping the diagonal
            # mask = jnp.expand_dims(~jnp.eye(n_electrons, dtype=bool), axis=(0, 3))
            # mask = jnp.repeat(jnp.repeat(mask, n_samples, axis=0), 3, axis=-1)
            # ee_vectors = ee_vectors[mask].reshape(n_samples, n_electrons ** 2 - n_electrons, 3)
    """
    """
    
    Notes:
        - for proof of this function go to debugging/drop_diagonal where compared with masking method
        - this removes the diagonal so a 3 x 3 matrix will give a 6 element vector
    """
    n = square.shape[0]
    split1 = jnp.split(square, n, axis=0)
    upper = [jnp.split(split1[i], [j], axis=1)[1] for i, j in zip(range(0, n), range(1, n))]
    lower = [jnp.split(split1[i], [j], axis=1)[0] for i, j in zip(range(1, n), range(1, n))]
    arr = [ls[i] for i in range(n-1) for ls in (upper, lower)]
    result = jnp.concatenate(arr, axis=1)
    return jnp.squeeze(result)


def linear_split(p: jnp.array,
                 data: jnp.array,
                 activations: list,
                 d0: jnp.array) -> jnp.array:

    activation = data
    activations.append(activation)
    pre_activation = jnp.dot(data, p) + d0
    return pre_activation


def linear(p: jnp.array,
           data: jnp.array,
           split: jnp.array,
           activations: list,
           d0: jnp.array) -> jnp.array:

    bias = jnp.ones((*data.shape[:-1], 1))
    activation = jnp.concatenate([data, bias], axis=-1)
    activations.append(activation)

    pre_activation = jnp.dot(activation, p) + d0
    return jnp.tanh(pre_activation + split)


def linear_pairwise(p: jnp.array,
                    data: jnp.array,
                    activations: list,
                    d0: jnp.array) -> jnp.array:

    bias = jnp.ones((*data.shape[:-1], 1))
    activation = jnp.concatenate([data, bias], axis=-1)
    activations.append(activation)

    pre_activation = jnp.dot(activation, p) + d0
    return jnp.tanh(pre_activation)


def env_linear_i(params: jnp.array,
                 data: jnp.array,
                 activations: list,
                 d0: jnp.array) -> jnp.array:
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
    pre_activations = jnp.matmul(activation, params) + d0  # (j, k * i)
    pre_activations = jnp.transpose(pre_activations).reshape(-1, n_spins, n_spins)  # (k, i, j)
    return pre_activations


def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(tensor.shape[axis], axis=axis)]