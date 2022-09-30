import jax.numpy as jnp
from jax import vmap
import numpy as np


def create_wf(mol, kfac=False):

    n_up, n_down, r_atoms, n_el = mol.n_up, mol.n_down, mol.r_atoms, mol.n_el
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph)

    def _wf_orbitals(params, walkers):

        if len(walkers.shape) == 1:  # this is a hack to get around the jvp
            walkers = walkers.reshape(n_up+n_down, 3)

        ae_vectors = compute_ae_vectors_i(walkers, r_atoms)

        single, pairwise = compute_inputs_i(walkers, ae_vectors)

        single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *masks[0])

        split = linear_split(params['split0'], split)
        single = linear(params['s0'], single_mixed, split)
        pairwise = linear_pairwise(params['p0'], pairwise)

        for (split_params, s_params, p_params), mask in zip(params['intermediate'], masks[1:]):

            single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *mask)

            split = linear_split(split_params, split)
            single = linear(s_params, single_mixed, split) + single
            pairwise = linear_pairwise(p_params, pairwise) + pairwise

        ae_up, ae_down = jnp.split(ae_vectors, [n_up], axis=0)
        data_up, data_down = jnp.split(single, [n_up], axis=0)

        factor_up = env_linear_i(params['envelopes']['linear'][0], data_up)
        factor_down = env_linear_i(params['envelopes']['linear'][1], data_down)

        exp_up = env_sigma_i(params['envelopes']['sigma'][0], ae_up)
        exp_down = env_sigma_i(params['envelopes']['sigma'][1], ae_down)

        orb_up = env_pi_i(params['envelopes']['pi'][0], factor_up, exp_up)
        orb_down = env_pi_i(params['envelopes']['pi'][1], factor_down, exp_down)
        return orb_up, orb_down

    def _wf(params, walkers):

        orb_up, orb_down = _wf_orbitals(params, walkers)
        log_psi = logabssumdet(orb_up, orb_down)
        return log_psi

    return _wf, _wf_orbitals


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


compute_ae_vectors = vmap(compute_ae_vectors_i, in_axes=(0, None))


def drop_diagonal_i(square):
    """
    for proof of this awesomeness go to debugging/drop_diagonal where compared with masking method
    """
    n = square.shape[0]
    split1 = jnp.split(square, n, axis=0)
    upper = [jnp.split(split1[i], [j], axis=1)[1] for i, j in zip(range(0, n), range(1, n))]
    lower = [jnp.split(split1[i], [j], axis=1)[0] for i, j in zip(range(1, n), range(1, n))]
    arr = [ls[i] for i in range(n - 1) for ls in (upper, lower)]
    result = jnp.concatenate(arr, axis=1)
    return jnp.squeeze(result)


# drop_diagonal = vmap(drop_diagonal_i, in_axes=(0,))


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

    re1 = jnp.expand_dims(walkers, axis=1)
    re2 = jnp.transpose(re1, [1, 0, 2])
    ee_vectors = re1 - re2
    # print(ee_vectors.shape)
    ee_vectors = drop_diagonal_i(ee_vectors)
    # print(ee_vectors.shape)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


compute_inputs = vmap(compute_inputs_i, in_axes=(0, 0))


def linear_split(p: jnp.array, data: jnp.array) -> jnp.array:
    return jnp.dot(data, p)


def linear(p: jnp.array,
           data: jnp.array,
           split: jnp.array) -> jnp.array:
    bias = jnp.ones((*data.shape[:-1], 1))
    data = jnp.concatenate([data, bias], axis=-1)
    return jnp.tanh(jnp.dot(data, p) + split)


def linear_pairwise(p: jnp.array,
                    data: jnp.array) -> jnp.array:
    bias = jnp.ones((*data.shape[:-1], 1))
    data = jnp.concatenate([data, bias], axis=-1)
    return jnp.tanh(jnp.dot(data, p))


def env_linear_i(params: jnp.array,
                 data: jnp.array) -> jnp.array:
    # params (k, i, f)
    # data (j, f)

    data = jnp.transpose(data)
    bias = jnp.ones((1, data.shape[-1]))
    data = jnp.concatenate((data, bias), axis=0)
    out = jnp.dot(params, data)

    # bias = jnp.ones((data.shape[0], 1))
    # data = jnp.concatenate((data, bias), axis=1)
    # out = jnp.einsum('jf,kif->kij', data, params)
    # print(out.shape)
    return out


env_linear = vmap(env_linear_i, in_axes=(None, 0))


def env_sigma_i(sigmas: jnp.array,
                ae_vectors: jnp.array) -> jnp.array:
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

    n_spin, n_atom, _ = ae_vectors.shape

    ae_vectors = [jnp.squeeze(x) for x in jnp.split(ae_vectors, n_atom, axis=1)]

    outs = []
    for ae_vector, sigma in zip(ae_vectors, sigmas):
        exponent = jnp.matmul(ae_vector, sigma).reshape(n_spin, 3, -1, n_spin, 1, order='F')
        out = jnp.exp(-jnp.linalg.norm(exponent, axis=1))
        outs.append(out)

    return jnp.concatenate(outs, axis=-1)


def env_pi_i(pi: jnp.array,
             factor: jnp.array,
             exponential: jnp.array) -> jnp.array:
    # exponential (j k i m)
    # factor (k i j)

    # orbitals = factor * jnp.einsum('jkim,kim->kij', exponential, pi)

    shape = exponential.shape
    exponential = exponential.reshape(shape[0], -1)
    orbitals = exponential * pi
    orbitals = orbitals.reshape(shape).sum(-1)

    # print(factor.shape, orbitals.shape)
    return factor * jnp.transpose(orbitals, (1, 2, 0))


env_pi = vmap(env_pi_i, in_axes=(None, 0, 0))


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
    # pairwise (n_samples, n_el, n_pairwise_features)
    # spin_up_mask = self.spin_up_mask.repeat((n_samples, 1, 1))
    # spin_down_mask = self.spin_down_mask.repeat((n_samples, 1, 1))

    # print(single.shape, pairwise.shape, single_up_mask.shape, pairwise_up_mask.shape)
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


mixer = vmap(mixer_i, in_axes=(0, 0, None, None, None, None, None, None, None))