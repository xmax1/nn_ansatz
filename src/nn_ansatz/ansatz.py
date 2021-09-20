from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np

from .utils import remove_aux
from .parameters import expand_d0s, initialise_d0s


def create_wf(mol, kfac: bool=False, orbitals: bool=False, signed: bool=False):
    ''' initializes the wave function ansatz for various applications '''

    print('creating wf')
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph, mol.n_in)

    _compute_exponents = create_compute_orbital_exponents(periodic_boundaries=mol.periodic_boundaries,
                                                          orbital_decay=mol.orbital_decay,
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

    # for (split_params, s_params, p_params), (split_per, s_per, p_per), mask \
    #         in zip(params['intermediate'], d0s['intermediate'], masks[1:]):
    #     single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *mask)

    #     split = linear_split(split_params, split, activations, split_per)
    #     single = linear(s_params, single_mixed, split, activations, s_per) + single
    #     pairwise = linear_pairwise(p_params, pairwise, activations, p_per) + pairwise

    for i, mask in enumerate(masks[1:], 1):
        single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *mask)

        split = linear_split(params['split%i'%i], split, activations, d0s['split%i'%i])
        single = linear(params['s%i'%i], single_mixed, split, activations, d0s['s%i'%i]) + single
        pairwise = linear_pairwise(params['p%i'%i], pairwise, activations, d0s['p%i'%i]) + pairwise

    ae_up, ae_down = jnp.split(ae_vectors, [mol.n_up], axis=0)
    data_up, data_down = jnp.split(single, [mol.n_up], axis=0)

    factor_up = env_linear_i(params['env_lin_up'], data_up, activations, d0s['env_lin_up'])
    factor_down = env_linear_i(params['env_lin_down'], data_down, activations, d0s['env_lin_down'])

    exp_up = _env_sigma_i(params['env_sigma_up'], ae_up, activations, d0s['env_sigma_up']) ##
    exp_down = _env_sigma_i(params['env_sigma_down'], ae_down, activations, d0s['env_sigma_down']) ##

    orb_up = env_pi_i(params['env_pi_up'], factor_up, exp_up, activations, d0s['env_pi_up'])
    orb_down = env_pi_i(params['env_pi_down'], factor_down, exp_down, activations, d0s['env_pi_down'])
    return orb_up, orb_down, activations


def compute_ae_vectors_i(walkers: jnp.array, r_atoms: jnp.array) -> jnp.array:
    ''' computes the nuclei-electron displacement vectors '''
    r_atoms = jnp.expand_dims(r_atoms, axis=0)
    walkers = jnp.expand_dims(walkers, axis=1)
    ae_vectors = r_atoms - walkers
    return ae_vectors


def compute_ae_vectors_periodic_i(walkers: jnp.array, r_atoms: jnp.array, unit_cell_length: float=1.) -> jnp.array:
    ''' computes the nuclei-electron displacement vectors under the minimum image convention '''
    r_atoms = jnp.expand_dims(r_atoms, axis=0)
    walkers = jnp.expand_dims(walkers, axis=1)
    ae_vectors = r_atoms - walkers
    ae_vectors = apply_minimum_image_convention(ae_vectors, unit_cell_length)
    return ae_vectors


def create_compute_inputs_i(mol):

    if mol.scalar_inputs:
        return compute_inputs_scalar_inputs_i

    if mol.periodic_boundaries:
        _compute_inputs_periodic_i = partial(compute_inputs_periodic_i, n_periodic_input=mol.n_periodic_input, unit_cell_length=mol.unit_cell_length)
        
        return _compute_inputs_periodic_i
    
    return compute_inputs_i


def compute_ee_vectors_i(walkers):
    ''' computes the electron-electron displacement vectors '''
    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(re1, [1, 0, 2])
    ee_vectors = re2 - re1
    return ee_vectors


def compute_inputs_i(walkers: jnp.array, ae_vectors: jnp.array):
    
    n_electrons, n_atoms = ae_vectors.shape[:2]

    ae_distances = jnp.linalg.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = jnp.concatenate([ae_vectors, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_electrons, 4 * n_atoms)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def compute_inputs_periodic_i(walkers, ae_vectors_min_im, n_periodic_input, unit_cell_length=1.):

    n_electrons, n_atoms = ae_vectors_min_im.shape[:2]

    ae_distances = jnp.linalg.norm(ae_vectors_min_im, axis=-1, keepdims=True)
    ae_vectors_periodic = jnp.concatenate([jnp.sin((2.*i*jnp.pi / unit_cell_length) * ae_vectors_min_im) for i in range(1, n_periodic_input+1)], axis=-1)
    single_inputs = jnp.concatenate([ae_vectors_periodic, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_electrons, ((n_periodic_input * 3) + 1) * n_atoms)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_vectors = apply_minimum_image_convention(ee_vectors, unit_cell_length)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    ee_vectors_periodic = jnp.concatenate([jnp.sin((2.*i*jnp.pi / unit_cell_length) * ee_vectors) for i in range(1, n_periodic_input+1)], axis=-1)
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


def compute_ee_vectors_no_grad_i(walkers, walkers_no_grad):
    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(walkers_no_grad[None, ...], [1, 0, 2])
    ee_vectors = re1 - re2
    return ee_vectors


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
    # params (k * i, f)
    # data (j, f)

    n_spins = data.shape[0]

    bias = jnp.ones((n_spins, 1))
    activation = jnp.concatenate((data, bias), axis=1)
    activations.append(activation)
    pre_activations = jnp.matmul(activation, params) + d0
    pre_activations = jnp.transpose(pre_activations).reshape(-1, n_spins, n_spins)
    return pre_activations


def split_and_squeeze(tensor, axis=0):
    return [x.squeeze(axis) for x in tensor.split(tensor.shape[axis], axis=axis)]


def anisotropic_exponent(ae_vector, sigma, d0, n_spin, unit_cell_length=1., periodic_boundaries=False, eps=0.0001):
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
    
    pre_activation = jnp.matmul(ae_vector, sigma)  + d0
    '''this line is required to force the situations where -jnp.inf + jnp.inf = jnp.nan creates a nan from the exponential and not a zero 
    (else jnp.inf + jnp.inf = jnp.inf)'''
    # pre_activation = jnp.where(jnp.isnan(pre_activation), jnp.inf, pre_activation)

    exponent = pre_activation.reshape(n_spin, 3, -1, n_spin, 1, order='F')
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
        exponential = exponential.reshape(n_spin, -1, n_spin, 1, order='F')

        # exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(1. - norm) * sigma + d0) - 2 * jnp.exp(-(1. / 2.) * sigma)
        # exponential = jnp.where(norm < ucl2, exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

        # tr_ae_vector = ae_vector.dot(inv_real_basis)
        # tr_norm = jnp.linalg.norm(tr_ae_vector, axis=-1)
        # exponential = jnp.where(tr_norm < min_cell_width / 2., exponential, jnp.zeros_like(exponential)).reshape(n_spin, -1, n_spin, 1, order='F')

    return norm, exponential


def env_sigma_i(sigmas: jnp.array,
                ae_vectors: jnp.array,
                activations: list,
                d0s: jnp.array,
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

    n_spin, n_atom, _ = ae_vectors.shape
    # ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    ae_vectors = split_and_squeeze(ae_vectors, 1)
    d0s = split_and_squeeze(d0s, 0)
    sigmas = split_and_squeeze(sigmas, 0)

    outs = []
    for ae_vector, sigma, d0 in zip(ae_vectors, sigmas, d0s):

        activation, exponential = _compute_exponents(ae_vector, sigma, d0, n_spin)

        activations.append(activation)
        
        outs.append(exponential)

    return jnp.concatenate(outs, axis=-1)


def create_compute_orbital_exponents(orbital_decay='anisotropic', 
                                     periodic_boundaries=False,
                                     unit_cell_length=1.):

    if orbital_decay == 'anisotropic':
        _compute_exponent = partial(anisotropic_exponent, periodic_boundaries=periodic_boundaries, unit_cell_length=unit_cell_length)
            
    elif orbital_decay == 'isotropic':
        _compute_exponent = partial(isotropic_exponent, periodic_boundaries=periodic_boundaries, unit_cell_length=unit_cell_length)
        
    return _compute_exponent




def env_pi_i(pis: jnp.array,
             factor: jnp.array,
             exponential: jnp.array,
             activations: list,
             d0s) -> jnp.array:
    # exponential (j k i m)
    # factor (k i j)

    n_spins, n_det = exponential.shape[:2]

    # EINSUM METHOD (does not work with kfac)
    # orbitals = factor * jnp.einsum('jkim,kim->kij', exponential, pi)

    # MATRIX METHOD (does not work with kfac, yet)
    # shape = exponential.shape
    # activation = exponential.reshape(shape[0], -1)
    # activations.append(activation)
    # pre_activations = activation * pi + d0
    # orbitals = pre_activations.reshape(shape).sum(-1)

    exponential = [jnp.squeeze(x, axis=(1, 2))
                   for y in jnp.split(exponential, n_spins, axis=2)
                   for x in jnp.split(y, n_det, axis=1)]  # n_det * n_spin of (n_spin, n_atom)

    [activations.append(x) for x in exponential]
    # [print((e @ pi).shape, d0.shape) for pi, e, d0 in zip(pis, exponential, d0s)]
    # pis = [pi for pi in pis.split(n_spins * n_det, axis=-1)]  # n_det * n_spin of pi (n_atom, )
    pis = split_and_squeeze(pis, -1) # n_det * n_spin of pi (n_atom, )
    d0s = split_and_squeeze(d0s, -1)  # d0s (n_spin, )

    orbitals = jnp.stack([(e @ pi) + d0 for pi, e, d0 in zip(pis, exponential, d0s)], axis=-1)
    # print(factor.shape, orbitals.shape)
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



# DEPRECIATED FUNCTIONS

def compute_ee_vectors_periodic_i_dep(walkers, real_basis, inv_real_basis):
    '''
    pseudocode:
        - translate to the unit cell 
        - compute the distances
        - element distances will be maximum 0.999 (as always in the same cell)
        - int(2 * element distances) will either be 0, 1 or -1
    '''
    unit_cell_walkers = walkers.dot(inv_real_basis)  # translate to the unit cell
    unit_cell_ee_vectors = compute_ee_vectors_i(unit_cell_walkers)
    min_image_unit_cell_ee_vectors = unit_cell_ee_vectors - (2 * unit_cell_ee_vectors).astype(int) * 1.  # 1 is length of unit cell put it here for clarity
    min_image_ee_vectors = min_image_unit_cell_ee_vectors.dot(real_basis)  # translate out of the unit cell
    return min_image_ee_vectors


def sigma_loopy_dep():
    # SIGMA LOOPY
    n_spins, n_atom, _ = ae_vectors.shape
    ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    m_layer = []
    for i, ae_vector in enumerate(ae_vectors):
        ki_layer = []
        for sigma, d0 in zip(sigmas[i], d0s[i]):
            activations.append(ae_vector)
            pre_activation = jnp.matmul(ae_vector, sigma) + d0
            out = jnp.exp(-jnp.linalg.norm(pre_activation, axis=1))
            ki_layer.append(out)
        m_layer.append(jnp.stack(ki_layer, axis=-1).reshape(n_spins, -1, n_spins))
    return jnp.stack(m_layer, axis=-1)
    return