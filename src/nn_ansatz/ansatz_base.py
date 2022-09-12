
from typing import Optional, Tuple
from .python_helpers import flatten

import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.nn import silu

eye_mask = lambda n_dim: (jnp.eye(n_dim)*-1)+1.


def no_af(x):
    return x


def snake(x):
    return jnp.sin(x)**2 + x


def compute_jastrow(rij: jnp.array, A: float, F: float):
    u = - (A / rij) * (1 - jnp.exp(-rij / F))
    return jnp.where(rij == 0., 0., u)  # sets the diagonal to zero and accounts for the limit as r_ij -> 0 for cuspss


def compute_djastrow(rij: float, A:float, F: float):
    return - ((- A / rij**2) * (1 - jnp.exp(-rij/F)) + (A/(rij*F)) * jnp.exp(-rij/F))


def compute_jastrow_arr(rij: jnp.array, A: float, F: jnp.array):
    u = - (A / rij) * (1. - jnp.exp(-rij / F))
    return jnp.where(rij == 0., 0., u)


def cubic(r: float):
    return [r**3, r**2, r, 1]


def cubic_arr(r: jnp.array, poly: jnp.array):
    return (poly[None, None, :] * jnp.stack([r**3, r**2, r, jnp.ones_like(r)], axis=-1)).sum(-1)


def dcubic(r: float):
    return [3*r**2, 2*r, 1, 0]



'''
floor = compute_jastrow(0.5, A, F)

    

    spline_fn = cubic_arr # if order == 3 # else raise NotImplementedError('Spline not implemented')

    def _compute_jastrow_factor_spline_i(walkers: jnp.array):
        
        ee_vectors = compute_ee_vectors_i(walkers) + 0.1 * jnp.eye(n_el)[..., None]
        ee_vectors = apply_minimum_image_convention(ee_vectors, basis, inv_basis)
        ee_distances = jnp.linalg.norm(ee_vectors, axis=-1)  # (n_el, n_el)

        jastrow = compute_jastrow_arr(ee_distances, A, F) # (n_el, n_el)

        
        
        return 0.5 * jastrow_spline.sum() # scalar



'''


def get_spline_polynomial(A:float, 
                          F:float, 
                          r_boundary: float, 
                          floor: float, 
                          order: int=3,
                          r_edge: float = 0.5):

    if order == 3:
        coefs = jnp.array([cubic(r_boundary),
                           dcubic(r_boundary),
                           cubic(r_edge),
                           dcubic(r_edge)])

        res = jnp.array([[compute_jastrow(r_boundary, A, F)],
            [compute_djastrow(r_boundary, A, F)],
            # [compute_jastrow(r_edge, A, F)],
            [0.0],
            [0.0]])

        poly_coefs = jnp.linalg.inv(coefs).dot(res)
    return poly_coefs.reshape(-1)


def create_jastrow_factor(n_el: int, 
                          n_up: int, 
                          volume: float,
                          density_parameter: float,
                          basis: jnp.array,
                          inv_basis: jnp.array,
                          r_boundary: float=0.25,
                          floor: float=0.,
                          order=3):

    n_down = n_el - n_up

    A = (density_parameter / 3.)**0.5 # factor of 2 for adjusting to the sin inputs
    # A = drop_diagonal_i(A)

    mask_up = jnp.concatenate([jnp.ones((n_up, n_up)), jnp.zeros((n_down, n_up))], axis=0)
    mask_down = jnp.concatenate([jnp.zeros((n_up, n_down)), jnp.ones((n_down, n_down))], axis=0)
    mask_same = jnp.concatenate([mask_up, mask_down], axis=1)
    mask_opp = (mask_same - 1.) * - 1.
    
    F_same = jnp.sqrt(jnp.pi * A)
    F_opp = jnp.sqrt(jnp.pi * A / 2.)
    F = mask_same * F_same + mask_opp * F_opp

    eye_shift = jnp.eye(n_el)[..., None]

    poly_same_coefs = get_spline_polynomial(A, F_same, r_boundary, floor, order=order)
    poly_opp_coefs = get_spline_polynomial(A, F_opp, r_boundary, floor, order=order)
    spline_fn = cubic_arr

    def _compute_jastrow_factor_i(params, walkers: jnp.array, activations, d0s):

        ee_vectors = compute_ee_vectors_i(walkers) + eye_shift
        ee_vectors = apply_minimum_image_convention(ee_vectors, basis, inv_basis)
        ee_vectors = jnp.sin(jnp.pi * ee_vectors) / 2.
        ee_distances = jnp.linalg.norm(ee_vectors, axis=-1) # (n_el, n_el)

        # split into up and down
        jastrow = compute_jastrow_arr(ee_distances, A, F) * eye_mask(n_el) # (n_el, n_el)

        # poly_same = spline_fn(ee_distances, poly_same_coefs)
        # poly_opp = spline_fn(ee_distances, poly_opp_coefs)
        
        # poly = mask_same * poly_same + mask_opp * poly_opp # (n_el, n_el)
        
        # jastrow = jnp.where(ee_distances > r_boundary, poly, jastrow) # (n_el, n_el)
        # jastrow = jnp.where(ee_distances > 0.5, 0.0, jastrow) # (n_el, n_el)
        # jastrow = jastrow * eye_mask

        # ee_f = ee_distances.reshape(-1, 1)
        # ee_f = jnp.concatenate([ee_f**i for i in range(1, 6)], axis=-1)
        # activations.append(ee_f)

        # fr = (ee_f @ params['jf_bf'] + d0s['jf_bf'])
        # G_i = (ee_vectors * eye_mask[..., None] * fr.reshape(n_el, n_el, 1)).sum(0)

        # jastrow -= (G_i * G_i).sum()

        return 0.5 * jastrow.sum()

    return _compute_jastrow_factor_i


def transform_vector_space(vectors: jnp.array, basis: jnp.array, on=False) -> jnp.array:
    '''
    case 1 catches non-orthorhombic cells 
    case 2 for orthorhombic and cubic cells
    '''
    if on:
        if basis.shape == (3, 3):
            return jnp.dot(vectors, basis)
        else:
            return vectors * basis
    else:
        return vectors


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
    return single_stream_vectors  # (n_el, n_atoms, 3)


def compute_ee_vectors_i(walkers):
    ''' computes the electron-electron displacement vectors '''
    # re1 = jnp.expand_dims(walkers, axis=1)
    # re2 = jnp.transpose(re1, [1, 0, 2])
    # ee_vectors = re2 - re1
    return walkers[None, ...] - walkers[:, None, ...]


def bowl(walkers_transformed: jnp.array, walkers: Optional[jnp.array]=None):
    if walkers is None:
        walkers = walkers_transformed
    return (walkers**2 / jnp.exp(4. * jnp.abs(walkers_transformed))) * (jnp.exp(2) / (2 * 0.5**2))


def input_activation(inputs: jnp.array, 
                     inv_basis: jnp.array, 
                     nonlinearity: str = 'sin',
                     kpoints: Optional[jnp.array] = None,
                     single_stream: bool = False):
    inputs_transformed = transform_vector_space(inputs, inv_basis)
    split = nonlinearity.split('+')
    
    if 'sin' in nonlinearity:
        sin_desc = [x for x in split if 'sin' in x][0]
        nsin = int(sin_desc[:-3]) if len(sin_desc) > 3 else 1
        sin_features = [jnp.sin(2.*i*jnp.pi*inputs_transformed) for i in range(1, nsin+1)]
    else:
        sin_features = []

    if 'cos' in nonlinearity:
        cos_desc = [x for x in split if 'cos' in x][0]
        ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1
        cos_features = [jnp.cos(2.*i*jnp.pi*inputs_transformed) for i in range(1, ncos+1)]
    else:
        cos_features = []

    if 'kpoints' in nonlinearity and single_stream:
        kpoints_desc = [x for x in split if 'kpoints' in x][0]
        nkpoints = int(kpoints_desc[:-7])
        sink = jnp.sin(inputs @ kpoints[1:nkpoints:2, :].T).mean(axis=0, keepdims=True).repeat(inputs.shape[0], axis=0)
        cosk = jnp.cos(inputs @ kpoints[2:nkpoints:2, :].T).mean(axis=0, keepdims=True).repeat(inputs.shape[0], axis=0)
        
        # cosk = jnp.cos(inputs @ kpoints[7:nkpoints:2, :].T)
        # sink = jnp.sin(inputs @ kpoints[8:nkpoints:2, :].T)

        rho_k = [sink, cosk]
    else:
        rho_k = []
    
    return jnp.concatenate([*sin_features, *cos_features, *rho_k], axis=-1)
    

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
                     pbc: bool=False,
                     input_activation_nonlinearity: str = 'sin',
                     kpoints: Optional[jnp.array] = None):

    n_el, n_features = single_stream_vectors.shape[:2]

    single_distances = [jnp.linalg.norm(single_stream_vectors, axis=-1, keepdims=True)]
    if pbc: 
        single_distances = []
        single_stream_vectors = input_activation(single_stream_vectors, inv_basis, nonlinearity=input_activation_nonlinearity, kpoints=kpoints, single_stream=True)

    single_inputs = jnp.concatenate([single_stream_vectors, *single_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_el, -1)

    if pbc: 
        ee_vectors = compute_ee_vectors_i(walkers)
        ee_vectors = apply_minimum_image_convention(ee_vectors, basis, inv_basis)
        ee_transformed = jnp.sin(jnp.pi*ee_vectors)/2. + jnp.eye(n_el)[..., None]
        ee_distances = [jnp.linalg.norm(ee_transformed, axis=-1, keepdims=True) * eye_mask(n_el)[..., None]]
        ee_vectors = input_activation(ee_vectors, inv_basis, nonlinearity=input_activation_nonlinearity)
    else:
        ee_vectors = compute_ee_vectors_i(walkers)
        ee_vectors = drop_diagonal_i(ee_vectors)
        ee_distances = [jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)]
    
    pairwise_inputs = jnp.concatenate([ee_vectors, *ee_distances], axis=-1)

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


def create_masks(n_electrons, n_up, n_layers):

    masks = [create_masks_layer(n_electrons, n_up)]

    for _ in range(n_layers):
        masks.append(create_masks_layer(n_electrons, n_up))

    masks.append(create_masks_layer(n_electrons, n_up))

    return masks


def create_masks_layer(n_electrons, n_up):
    # pairwise spin masks
    ups = np.ones(n_electrons)
    ups[n_up:] = 0.
    downs = (ups-1.)*-1.

    pairwise_up_mask = []
    pairwise_down_mask = []
    mask = np.zeros((n_electrons, n_electrons))

    for electron in range(n_electrons):
        mask_up = mask.copy()
        mask_up[electron, :] = ups
        pairwise_up_mask.append(mask_up)
        # mask_up = mask_up[eye_mask].reshape(-1) # for when drop diagonal enforced

        mask_down = mask.copy()
        mask_down[electron, :] = downs
        pairwise_down_mask.append(mask_down)

    pairwise_up_mask = jnp.stack(pairwise_up_mask, axis=0)[..., None]
    pairwise_down_mask = jnp.stack(pairwise_down_mask, axis=0)[..., None]

    return pairwise_up_mask, pairwise_down_mask


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
           residual: jnp.array,
           nonlinearity: str = 'tanh') -> jnp.array:

    bias = jnp.ones((*data.shape[:-1], 1))
    activation = jnp.concatenate([data, bias], axis=-1)
    activations.append(activation)

    pre_activation = jnp.dot(activation, p) + split + d0
    data = layer_activation(pre_activation, nonlinearity=nonlinearity)

    if residual.shape == pre_activation.shape:
        data += residual
    return data
    
    
def layer_activation(pre_activation, nonlinearity='cos'):
    if nonlinearity is None:
        return pre_activation
    if nonlinearity == 'tanh':
        return jnp.tanh(pre_activation)
    elif nonlinearity == 'sin':
        return jnp.sin(pre_activation)
    elif nonlinearity == 'cos':
        return jnp.cos(pre_activation)
    elif nonlinearity == 'silu':
        return silu(pre_activation)
    elif nonlinearity == 'snake':
        return snake(pre_activation)
    elif nonlinearity == 'no_af':
        return pre_activation
    else:
        exit('nonlinearity not available')


def separate_same_and_diff(data, n_up, n_down):
    # data (x, x, f)
    if n_down == 0:
        return data, None
    else:
        data_t, data_b = data.split([n_up], axis=0)
        data_tl, data_tr = data_t.split([n_up], axis=1)
        data_bl, data_br = data_b.split([n_up], axis=1)
        data_same = jnp.concatenate([data_tl.reshape(n_up**2, -1), data_br.reshape(n_down**2, -1)], axis=0)
        data_diff = jnp.concatenate([data_tr.reshape(n_up*n_down, -1), data_bl.reshape(n_up*n_down, -1)], axis=0)
        return data_same, data_diff


def join_same_and_diff(same, diff, n_up, n_down):
    if n_down == 0:
        return same
    else:
        data_tl, data_br = same.split([n_up**2], axis=0)
        data_tl = data_tl.reshape(n_up, n_up, -1)
        data_br = data_br.reshape(n_down, n_down, -1)

        data_tr, data_bl = diff.split([n_up*n_down], axis=0)
        data_tr = data_tr.reshape(n_up, n_down, -1)
        data_bl = data_bl.reshape(n_down, n_up, -1)

        data_t = jnp.concatenate([data_tl, data_tr], axis=1)
        data_b = jnp.concatenate([data_bl, data_br], axis=1)
        return jnp.concatenate([data_t, data_b], axis=0)



def linear_pairwise(params,
                    layer: int,
                    data: jnp.array,
                    activations: list,
                    d0s,
                    residual: jnp.array,
                    n_up: int,
                    n_down: int,
                    psplit_spins: bool,
                    nonlinearity: str = 'tanh') -> jnp.array:
    n_el = n_up + n_down

    if psplit_spins:                              
        data_same, data_diff = separate_same_and_diff(data, n_up, n_down)

        bias = jnp.ones((*data_same.shape[:-1], 1))
        a_same = jnp.concatenate([data_same, bias], axis=-1)
        activations.append(a_same)
        
        bias = jnp.ones((*data_diff.shape[:-1], 1))
        a_diff = jnp.concatenate([data_diff, bias], axis=-1)
        activations.append(a_diff)
        
        pre_same = layer_activation(jnp.dot(a_same, params['ps%i'%layer]) + d0s['ps%i'%layer], nonlinearity=nonlinearity)
        pre_diff = layer_activation(jnp.dot(a_diff, params['pd%i'%layer]) + d0s['pd%i'%layer], nonlinearity=nonlinearity)

        pairwise = join_same_and_diff(pre_same, pre_diff, n_up, n_down)
    else:
        data = data.reshape(n_el**2, -1)
        bias = jnp.ones((*data.shape[:-1], 1))
        a_same = jnp.concatenate([data, bias], axis=-1)
        activations.append(a_same)

        pairwise = layer_activation(jnp.dot(a_same, params['ps%i'%layer]) + d0s['ps%i'%layer], nonlinearity=nonlinearity)

        pairwise = pairwise.reshape(n_el, n_el, -1)

    if pairwise.shape == residual.shape:
        pairwise += residual
    
    return pairwise


def env_linear_i(params: jnp.array,
                 data: jnp.array,
                 activations: list,
                 d0: jnp.array,
                 nonlinearity=None) -> jnp.array:
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
    pre_activations = jnp.matmul(activation, params) + d0  # (j, i)
    if nonlinearity is not None:
        pre_activations = layer_activation(pre_activations, nonlinearity=nonlinearity)
    pre_activations = jnp.transpose(pre_activations) # (i j)
    return pre_activations


def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(tensor.shape[axis], axis=axis)]


