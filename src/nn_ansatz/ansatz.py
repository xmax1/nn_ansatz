from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax import vmap, lax
import numpy as np

from .utils import remove_aux
from .parameters import expand_d0s, initialise_d0s


def create_wf(mol, kfac: bool=False, orbitals: bool=False):
    ''' initializes the wave function ansatz for various applications '''

    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph)

    if mol.periodic_boundaries:
        _env_sigma_i = env_sigma_i_periodic

    _compute_inputs_i = create_compute_inputs_i(mol)

    _wf_orbitals = partial(wf_orbitals, mol=mol, masks=masks, compute_inputs_i=_compute_inputs_i, env_sigma_i=_env_sigma_i)

    def _wf(params, walkers, d0s):
        orb_up, orb_down, _ = _wf_orbitals(params, walkers, d0s)
        log_psi = logabssumdet(orb_up, orb_down)
        return log_psi

    def _kfac_wf(params, walkers, d0s):
        orb_up, orb_down, activations = _wf_orbitals(params, walkers, d0s)
        log_psi = logabssumdet(orb_up, orb_down)
        return log_psi, activations
    
    d0s = initialise_d0s(mol)

    if orbitals:
        _wf_orbitals_remove_activations = remove_aux(_wf_orbitals, axis=1)
        return partial(_wf_orbitals_remove_activations, d0s=d0s)

    if kfac:
        return vmap(_kfac_wf, in_axes=(None, 0, 0))

    _partial_wf = partial(_wf, d0s=d0s)
    _vwf = vmap(_partial_wf, in_axes=(None, 0))
    
    return _vwf


def wf_orbitals(params, 
                walkers, 
                d0s, 
                mol, 
                masks,
                compute_inputs_i: Callable, 
                env_sigma_i: Callable):

    if len(walkers.shape) == 1:  # this is a hack to get around the jvp
        walkers = walkers.reshape(mol.n_up + mol.n_down, 3)

    activations = []

    ae_vectors = compute_ae_vectors_i(walkers, mol.r_atoms)

    single, pairwise = compute_inputs_i(walkers, ae_vectors)

    single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *masks[0])

    split = linear_split(params['split0'], split, activations, d0s['split0'])
    single = linear(params['s0'], single_mixed, split, activations, d0s['s0'])
    pairwise = linear_pairwise(params['p0'], pairwise, activations, d0s['p0'])

    for (split_params, s_params, p_params), (split_per, s_per, p_per), mask \
            in zip(params['intermediate'], d0s['intermediate'], masks[1:]):
        single_mixed, split = mixer_i(single, pairwise, mol.n_el, mol.n_up, mol.n_down, *mask)

        split = linear_split(split_params, split, activations, split_per)
        single = linear(s_params, single_mixed, split, activations, s_per) + single
        pairwise = linear_pairwise(p_params, pairwise, activations, p_per) + pairwise

    ae_up, ae_down = jnp.split(ae_vectors, [mol.n_up], axis=0)
    data_up, data_down = jnp.split(single, [mol.n_up], axis=0)

    factor_up = env_linear_i(params['envelopes']['linear'][0], data_up, activations, d0s['envelopes']['linear'][0])
    factor_down = env_linear_i(params['envelopes']['linear'][1], data_down, activations, d0s['envelopes']['linear'][1])

    exp_up = env_sigma_i(params['envelopes']['sigma']['up'], ae_up, activations, d0s['envelopes']['sigma']['up'], mol.min_cell_width)
    exp_down = env_sigma_i(params['envelopes']['sigma']['down'], ae_down, activations, d0s['envelopes']['sigma']['down'], mol.min_cell_width)

    orb_up = env_pi_i(params['envelopes']['pi'][0], factor_up, exp_up, activations, d0s['envelopes']['pi'][0])
    orb_down = env_pi_i(params['envelopes']['pi'][1], factor_down, exp_down, activations, d0s['envelopes']['pi'][1])
    return orb_up, orb_down, activations


def create_compute_inputs_i(mol):

    if mol.periodic_boundaries:
        _compute_inputs_periodic_i = partial(compute_inputs_periodic_i, 
                                         min_cell_width=mol.min_cell_width,
                                         real_basis=mol.real_basis, 
                                         inv_real_basis=mol.inv_real_basis)
        return _compute_inputs_periodic_i
    
    return compute_inputs_i


def compute_inputs_i(walkers, ae_vectors):
    """
    Notes:
        Previous masking code for dropping the diagonal
            # mask = jnp.expand_dims(~jnp.eye(n_electrons, dtype=bool), axis=(0, 3))
            # mask = jnp.repeat(jnp.repeat(mask, n_samples, axis=0), 3, axis=-1)
            # ee_vectors = ee_vectors[mask].reshape(n_samples, n_electrons ** 2 - n_electrons, 3)
    """
    n_electrons, n_atoms = ae_vectors.shape[:2]

    ae_distances = jnp.linalg.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = jnp.concatenate([ae_vectors, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_electrons, 4 * n_atoms)

    ee_vectors = compute_ee_vectors_i(walkers)
    ee_vectors = drop_diagonal_i(ee_vectors)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def compute_ee_vectors_i(walkers):
    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(re1, [1, 0, 2])
    ee_vectors = re1 - re2
    return ee_vectors


def compute_inputs_periodic_i(walkers, ae_vectors, min_cell_width, real_basis, inv_real_basis):
        """
        Notes:
            Previous masking code for dropping the diagonal
                # mask = jnp.expand_dims(~jnp.eye(n_electrons, dtype=bool), axis=(0, 3))
                # mask = jnp.repeat(jnp.repeat(mask, n_samples, axis=0), 3, axis=-1)
                # ee_vectors = ee_vectors[mask].reshape(n_samples, n_electrons ** 2 - n_electrons, 3)
        """
        n_electrons, n_atoms = ae_vectors.shape[:2]

        ae_distances = jnp.linalg.norm(ae_vectors, axis=-1, keepdims=True)
        ae_vectors_periodic = jnp.sin((jnp.pi / min_cell_width) * ae_vectors)
        single_inputs = jnp.concatenate([ae_vectors_periodic, ae_distances], axis=-1)
        single_inputs = single_inputs.reshape(n_electrons, 4 * n_atoms)

        ee_vectors = compute_ee_vectors_periodic_i(walkers, real_basis, inv_real_basis)
        ee_vectors = drop_diagonal_i(ee_vectors)
        ee_vectors_periodic = jnp.sin((jnp.pi / min_cell_width) * ee_vectors)
        ee_distances = jnp.linalg.norm(ee_vectors_periodic, axis=-1, keepdims=True)
        pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)
        return single_inputs, pairwise_inputs


def compute_ee_vectors_periodic_i(walkers, real_basis, inv_real_basis):
    unit_cell_walkers = walkers.dot(inv_real_basis)  # translate to the unit cell
    unit_cell_ee_vectors = compute_ee_vectors_i(unit_cell_walkers)
    min_image_unit_cell_ee_vectors = unit_cell_ee_vectors - (2 * unit_cell_ee_vectors).astype(int) * 1.  # 1 is length of unit cell put it here for clarity
    min_image_ee_vectors = min_image_unit_cell_ee_vectors.dot(real_basis)  # translate out of the unit cell
    return min_image_ee_vectors


def create_masks(n_atom, n_electrons, n_up, n_layers, n_sh, n_ph):

    n_sh_in, n_ph_in = 4 * n_atom, 4

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


def compute_ae_vectors_i(walkers: jnp.array,
                         r_atoms: jnp.array) -> jnp.array:
    r_atoms = jnp.expand_dims(r_atoms, axis=0)
    walkers = jnp.expand_dims(walkers, axis=1)
    ae_vectors = walkers - r_atoms
    return ae_vectors


def drop_diagonal_i(square):
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
    # params (k * i, f)
    # data (j, f)

    n_spins = data.shape[0]

    # data = jnp.transpose(data)
    # bias = jnp.ones((1, data.shape[-1]))
    # data = jnp.concatenate((data, bias), axis=0)
    # pre_activations = jnp.dot(params, data)

    bias = jnp.ones((n_spins, 1))
    activation = jnp.concatenate((data, bias), axis=1)
    activations.append(activation)
    pre_activations = jnp.matmul(activation, params) + d0
    pre_activations = jnp.transpose(pre_activations).reshape(-1, n_spins, n_spins)

    # bias = jnp.ones((data.shape[0], 1))
    # data = jnp.concatenate((data, bias), axis=1)
    # out = jnp.einsum('jf,kif->kij', data, params)
    # print(out.shape)
    # print(params.shape, pre_activations.shape, activation.shape)
    return pre_activations


env_linear = vmap(env_linear_i, in_axes=(None, 0))


def env_sigma_i(sigmas: jnp.array,
                ae_vectors: jnp.array,
                activations: list,
                d0s: jnp.array,
                min_cell_width: float) -> jnp.array:
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
    """
    # sigma (n_det, n_spin, n_atom, 3, 3)
    # ae_vectors (n_spin, n_atom, 3)

    # exponent = jnp.einsum('jmv,kimvc->jkimc', ae_vectors, sigma)
    # return jnp.exp(-jnp.linalg.norm(exponent, axis=-1))

    # SIGMA BROADCAST VERSION
    n_spin, n_atom, _ = ae_vectors.shape
    ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    outs = []
    for ae_vector, sigma, d0 in zip(ae_vectors, sigmas, d0s):
        activations.append(ae_vector)
        
        pre_activation = jnp.matmul(ae_vector, sigma) + d0
        exponent = pre_activation.reshape(n_spin, 3, -1, n_spin, 1, order='F')
        exponent = jnp.linalg.norm(exponent, axis=1)
        out = jnp.exp(-exponent)

        outs.append(out)
    return jnp.concatenate(outs, axis=-1)

    # SIGMA LOOPY
    # n_spins, n_atom, _ = ae_vectors.shape
    # ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    # m_layer = []
    # for i, ae_vector in enumerate(ae_vectors):
    #     ki_layer = []
    #     for sigma, d0 in zip(sigmas[i], d0s[i]):
    #         activations.append(ae_vector)
    #         pre_activation = jnp.matmul(ae_vector, sigma) + d0
    #         out = jnp.exp(-jnp.linalg.norm(pre_activation, axis=1))
    #         ki_layer.append(out)
    #     m_layer.append(jnp.stack(ki_layer, axis=-1).reshape(n_spins, -1, n_spins))
    # return jnp.stack(m_layer, axis=-1)


def env_sigma_i_periodic(sigmas: jnp.array,
                ae_vectors: jnp.array,
                activations: list,
                d0s: jnp.array,
                min_cell_width: float) -> jnp.array:
    
    boundary = min_cell_width / 2.
    # sigma (n_det, n_spin, n_atom, 3, 3)
    # ae_vectors (n_spin, n_atom, 3)

    # exponent = jnp.einsum('jmv,kimvc->jkimc', ae_vectors, sigma)
    # return jnp.exp(-jnp.linalg.norm(exponent, axis=-1))

    # SIGMA BROADCAST VERSION
    n_spin, n_atom, _ = ae_vectors.shape
    ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]
    outs = []
    for ae_vector, sigma, d0 in zip(ae_vectors, sigmas, d0s):

        # fix to half if below 
        ae_vector = jnp.where(ae_vector < -boundary, -boundary, ae_vector)
        ae_vector = jnp.where(ae_vector > boundary, boundary, ae_vector)

        # apply case wise functions
        ae_vector = jnp.where(ae_vector < -boundary/2., -1./(8.*(min_cell_width + 2.*ae_vector)), ae_vector)
        ae_vector = jnp.where(ae_vector > boundary/2., 1./(8.*(min_cell_width - 2.*ae_vector)), ae_vector)

        activations.append(ae_vector)

        pre_activation = jnp.matmul(ae_vector, sigma) + d0

        exponent = jnp.linalg.norm(pre_activation.reshape(n_spin, 3, -1, n_spin, 1, order='F'), axis=1)
        out = jnp.exp(-exponent)

        # out = jnp.exp(-r) + jnp.exp(-(min_cell_width - r)) - 2 * jnp.exp(-min_cell_width / 2.)
        # out = jnp.exp(-exponent) + jnp.exp(-(min_cell_width - exponent)) - 2 * jnp.exp(-min_cell_width / 2.)
        # out = jnp.where(exponent < min_cell_width / 2., out, jnp.zeros_like(out))
        outs.append(out)
    return jnp.concatenate(outs, axis=-1)


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
                   for x in jnp.split(y, n_det, axis=1)]

    [activations.append(x) for x in exponential]
    # [print((e @ pi).shape, d0.shape) for pi, e, d0 in zip(pis, exponential, d0s)]
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

    return jnp.log(jnp.abs(jnp.sum(argument, axis=0))) + logdet_max


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