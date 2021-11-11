
from typing import Callable, Optional
from functools import partial

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np


def transform_vector_space(vectors: jnp.array, basis: jnp.array) -> jnp.array:
    '''
    case 1 catches non-orthorhombic cells 
    case 2 for orthorhombic and cubic cells
    '''
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis


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
        if pbc: single_stream_vectors = apply_minimum_image_convention(single_stream_vectors, basis, inv_basis)
    return single_stream_vectors  # (n_el, n_atoms, 3)


def compute_ee_vectors_i(walkers):
    ''' computes the electron-electron displacement vectors '''
    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(re1, [1, 0, 2])
    ee_vectors = re2 - re1
    return ee_vectors


def input_activation_test(nonlinearity: str = 'sin', n_el=7, nf=3):
    split = nonlinearity.split('+')
    if 'bowl' in nonlinearity:
        bowl_features = [jnp.ones((n_el, nf))]
    else:
        bowl_features = []
    if 'sin' in nonlinearity:
        sin_desc = [x for x in split if 'sin' in x][0]
        nsin = int(sin_desc[:-3]) if len(sin_desc) > 3 else 1
        sin_features = [jnp.sin(2.*i*jnp.pi*jnp.ones((n_el, nf))) for i in range(1, nsin+1)]
    else:
        sin_features = []
    if 'cos' in nonlinearity:
        cos_desc = [x for x in split if 'sin' in x][0]
        ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1
        cos_features = [jnp.cos(2.*i*jnp.pi*jnp.ones((n_el, nf))) for i in range(1, ncos+1)]
    else:
        cos_features = []
    return jnp.concatenate([*sin_features, *cos_features, *bowl_features], axis=-1)


def input_activation(inputs: jnp.array, inv_basis: jnp.array, nonlinearity: str = 'sin'):
    inputs_transformed = transform_vector_space(inputs, inv_basis)
    split = nonlinearity.split('+')
    if 'bowl' in nonlinearity:
        bowl_features = [inputs**2 / jnp.exp(4. * jnp.abs(inputs_transformed))]
    else:
        bowl_features = []
    
    if 'sin' in nonlinearity:
        sin_desc = [x for x in split if 'sin' in x][0]
        nsin = int(sin_desc[:-3]) if len(sin_desc[0]) > 3 else 1
        sin_features = [jnp.sin(2.*i*jnp.pi*inputs_transformed) for i in range(1, nsin+1)]
    else:
        sin_features = []

    if 'cos' in nonlinearity:
        cos_desc = [x for x in split if 'cos' in x][0]
        ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1
        cos_features = [jnp.cos(2.*i*jnp.pi*inputs_transformed) for i in range(1, ncos+1)]
    else:
        cos_features = []
    
    return jnp.concatenate([*sin_features, *cos_features, *bowl_features], axis=-1)
    
    # elif nonlinearity == 'bowl':
    #     return 
    # elif nonlinearity == 'cos':
    #     return jnp.sin(2.*jnp.pi*inputs_transformed)
    # elif nonlinearity == 'sin+cos':
    #     return jnp.concatenate([jnp.sin(2.*jnp.pi*inputs_transformed), jnp.cos(2.*jnp.pi*inputs_transformed)], axis=-1)
    # elif 'sin' in nonlinearity and 'cos' in nonlinearity:
    #     fs = nonlinearity.split('+')
        
        
    #     sins = jnp.concatenate([jnp.sin(2.*i*jnp.pi*inputs_transformed) for i in range(1, nsin+1)], axis=-1)
    #     coss = jnp.concatenate([jnp.cos(2.*i*jnp.pi*inputs_transformed) for i in range(1, ncos+1)], axis=-1)
    #     if 'bowl' in nonlinearity:
    #         bowl = inputs**2 / jnp.exp(4. * jnp.abs(inputs_transformed))
    #         return jnp.concatenate([sins, coss, bowl], axis=-1)
    #     return jnp.concatenate([sins, coss], axis=-1)
    


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
    displacement_vectors = displacement_vectors - lax.stop_gradient(displace) 
    return displacement_vectors


def compute_inputs_i(walkers: jnp.array, 
                     single_stream_vectors: jnp.array, 
                     basis: Optional[jnp.array]=None,
                     inv_basis: Optional[jnp.array]=None,
                     pbc: bool=False,
                     input_activation_nonlinearity: str = 'sin'):

    n_el, n_features = single_stream_vectors.shape[:2]

    single_distances = jnp.linalg.norm(single_stream_vectors, axis=-1, keepdims=True)
    if pbc: 
        single_distances = jnp.linalg.norm(input_activation(single_stream_vectors, inv_basis, nonlinearity='bowl'), axis=-1, keepdims=True)

        single_stream_vectors = input_activation(single_stream_vectors, inv_basis, nonlinearity=input_activation_nonlinearity)

    single_inputs = jnp.concatenate([single_stream_vectors, single_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_el, -1)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    if pbc: 
        ee_vectors = apply_minimum_image_convention(ee_vectors, basis, inv_basis)

        ee_distances = jnp.linalg.norm(input_activation(ee_vectors, inv_basis, nonlinearity='bowl'), axis=-1, keepdims=True)

        ee_vectors = input_activation(ee_vectors, inv_basis, nonlinearity=input_activation_nonlinearity)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

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




def create_masks(n_electrons, n_up, n_layers, n_sh, n_ph, n_sh_in, n_ph_in):

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
           d0: jnp.array,
           nonlinearity: str = 'tanh') -> jnp.array:

    bias = jnp.ones((*data.shape[:-1], 1))
    activation = jnp.concatenate([data, bias], axis=-1)
    activations.append(activation)

    pre_activation = jnp.dot(activation, p) + d0
    if nonlinearity == 'tanh':
        return jnp.tanh(pre_activation + split)
    elif nonlinearity == 'sin':
        return jnp.sin(pre_activation + split)
    else:
        exit('nonlinearity not available')


def linear_pairwise(p: jnp.array,
                    data: jnp.array,
                    activations: list,
                    d0: jnp.array,
                    nonlinearity: str = 'tanh') -> jnp.array:

    bias = jnp.ones((*data.shape[:-1], 1))
    activation = jnp.concatenate([data, bias], axis=-1)
    activations.append(activation)

    pre_activation = jnp.dot(activation, p) + d0
    if nonlinearity == 'tanh':
        return jnp.tanh(pre_activation)
    elif nonlinearity == 'sin':
        return jnp.sin(pre_activation)
    else:
        exit('nonlinearity not available')


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
    pre_activations = pre_activations.reshape(n_spins, -1, n_spins)  # (j, k, i), env_pi reshapes the same way so k -> k
    pre_activations = jnp.transpose(pre_activations, (1, 2, 0)) # (k i j)
    return pre_activations


def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(tensor.shape[axis], axis=axis)]
