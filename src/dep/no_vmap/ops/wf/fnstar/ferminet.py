import jax.numpy as jnp
import numpy as np


def model(params,
          r_electrons,
          r_atoms = None,
          masks = None,
          n_up = None,
          n_down = None):
    n_samples = r_electrons.shape[0]
    if len(r_electrons.shape) == 2:  # this is a hack to get around the jvp
        r_electrons = r_electrons.reshape(n_samples, n_up+n_down, 3)

    n_samples, n_electrons = r_electrons.shape[:2]
    n_atoms = r_atoms.shape[1]

    # ae_vectors
    ae_vectors = compute_ae_vectors(r_electrons, r_atoms)

    # compute the inputs
    single, pairwise = compute_inputs(r_electrons, ae_vectors)
    # print(single.shape, pairwise.shape)

    # mix the inputs
    # print([x.shape for x in masks[0]])
    single, split = mixer(single, pairwise, n_electrons, n_up, n_down, *masks[0])

    # initial streams s0 and p0
    # print(single.shape, pairwise.shape, params['s0'].shape, params['p0'].shape)
    split = linear_split(params['split0'], split)
    single = linear(params['s0'], single, split)
    pairwise = linear_pairwise(params['p0'], pairwise)

    # intermediate layers including mix
    for (s_params, split_params, p_params), mask in zip(params['intermediate'], masks[1:]):
        # print(s_params.shape, p_params.shape, [x.shape for x in mask])

        single_mixed, split = mixer(single, pairwise, n_electrons, n_up, n_down, *mask)
        # print(single_mixed.shape)

        split = linear_split(split_params, split)
        single = linear(s_params, single_mixed, split) + single
        pairwise = linear_pairwise(p_params, pairwise) + pairwise
        # print(single.shape, pairwise.shape)

    # split
    ae_up, ae_down = jnp.split(ae_vectors, [n_up], axis=1)
    data_up, data_down = jnp.split(single, [n_up], axis=1)

    # envelopes
    # linear
    factor_up = env_linear(params['envelopes']['linear'][0], data_up)
    factor_down = env_linear(params['envelopes']['linear'][1], data_down)

    # sigma
    exp_up = env_sigma(params['envelopes']['sigma'][0], ae_up)
    exp_down = env_sigma(params['envelopes']['sigma'][1], ae_down)

    # pi
    # print(exp_up.shape, exp_down.shape, params['envelopes']['pi'][0].shape, params['envelopes']['pi'][1].shape)
    orb_up = env_pi(params['envelopes']['pi'][0], factor_up, exp_up)
    orb_down = env_pi(params['envelopes']['pi'][1], factor_down, exp_down)

    # logabssumdet
    # print(orb_up.shape, orb_down.shape)
    log_psi = logabssumdet(orb_up, orb_down)

    return log_psi


def create_masks(n_atom, n_electrons, n_up, n_layers, n_sh, n_ph):
    n_sh_in = 4 * n_atom
    n_ph_in = 4

    masks = [create_masks_layer(n_sh_in, n_ph_in, n_electrons, n_up)]

    for i in range(n_layers):
        masks.append(create_masks_layer(n_sh, n_ph, n_electrons, n_up))

    return masks


def create_masks_layer(n_sh, n_ph, n_electrons, n_up):
    # single spin masks
    eye_mask = ~np.eye(n_electrons, dtype=bool)
    n_down = n_electrons - n_up
    n_pairwise = n_electrons ** 2 - n_electrons

    tmp1 = jnp.ones((1, n_up, n_sh))
    tmp2 = jnp.zeros((1, n_down, n_sh))
    single_up_mask = jnp.concatenate((tmp1, tmp2), axis=1)
    single_down_mask = (jnp.concatenate((tmp1, tmp2), axis=1) - 1.) * -1.

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

    pairwise_up_mask = jnp.array(pairwise_up_mask).reshape((1, n_electrons, n_pairwise, 1))
    pairwise_up_mask = jnp.repeat(pairwise_up_mask, n_ph, axis=-1)

    pairwise_down_mask = jnp.array(pairwise_down_mask).reshape((1, n_electrons, n_pairwise, 1))
    pairwise_down_mask = jnp.repeat(pairwise_down_mask, n_ph, axis=-1)
    return single_up_mask, single_down_mask, pairwise_up_mask, pairwise_down_mask


def compute_ae_vectors(r_electrons: jnp.array,
                       r_atoms: jnp.array) -> jnp.array:
    r_atoms = jnp.expand_dims(r_atoms, axis=1)
    r_electrons = jnp.expand_dims(r_electrons, axis=2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


def drop_diagonal(square):
    """
    for proof of this awesomeness go to debugging/drop_diagonal where compared with masking method
    """
    n = square.shape[1]
    split1 = jnp.split(square, n, axis=1)
    upper = [jnp.split(split1[i], [j], axis=2)[1] for i, j in zip(range(0, n), range(1, n))]
    lower = [jnp.split(split1[i], [j], axis=2)[0] for i, j in zip(range(1, n), range(1, n))]
    arr = [ls[i] for i in range(n-1) for ls in (upper, lower)]
    result = jnp.concatenate(arr, axis=2)
    return jnp.squeeze(result)


def compute_inputs(r_electrons, ae_vectors):
    """
    Notes:
        Previous masking code for dropping the diagonal
            # mask = jnp.expand_dims(~jnp.eye(n_electrons, dtype=bool), axis=(0, 3))
            # mask = jnp.repeat(jnp.repeat(mask, n_samples, axis=0), 3, axis=-1)
            # ee_vectors = ee_vectors[mask].reshape(n_samples, n_electrons ** 2 - n_electrons, 3)
    """
    n_samples, n_electrons, n_atoms = ae_vectors.shape[:3]

    ae_distances = jnp.linalg.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = jnp.concatenate([ae_vectors, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_samples, n_electrons, 4 * n_atoms)

    re1 = jnp.expand_dims(r_electrons, axis=2)
    re2 = jnp.transpose(re1, [0, 2, 1, 3])
    ee_vectors = re1 - re2
    ee_vectors = drop_diagonal(ee_vectors)
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)

    return single_inputs, pairwise_inputs


def linear_split(p: jnp.array,
                 data: jnp.array) -> jnp.array:
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


def env_linear(params: jnp.array,
               data: jnp.array) -> jnp.array:
    bias = jnp.ones((*data.shape[:-1], 1))
    data = jnp.concatenate((data, bias), axis=-1)

    return jnp.einsum('njf,kif->njki', data, params)


def env_sigma(sigma: jnp.array,
              ae_vectors: jnp.array) -> jnp.array:
    exponent = jnp.einsum('njmv,kimvc->njkimc', ae_vectors, sigma)
    return jnp.exp(-jnp.linalg.norm(exponent, axis=-1))


def env_pi(pi: jnp.array,
           factor: jnp.array,
           exponential: jnp.array) -> jnp.array:
    orbitals = factor * jnp.einsum('njkim,kim->njki', exponential, pi)
    return jnp.transpose(orbitals, [0, 2, 1, 3])


def logabssumdet(orb_up: jnp.array,
                 orb_down: jnp.array) -> jnp.array:
    s_up, log_up = jnp.linalg.slogdet(orb_up)
    s_down, log_down = jnp.linalg.slogdet(orb_down)

    logdet_sum = log_up + log_down
    logdet_max = jnp.max(logdet_sum)

    argument = s_up * s_down * jnp.exp(logdet_sum - logdet_max)

    return jnp.log(jnp.abs(jnp.sum(argument, axis=1))) + logdet_max


def mixer(single: jnp.array,
          pairwise: jnp.array,
          n_electrons,
          n_up,
          n_down,
          single_up_mask,
          single_down_mask,
          pairwise_up_mask,
          pairwise_down_mask):
    # single (n_samples, n_electrons, n_single_features)
    # pairwise (n_samples, n_electrons, n_pairwise_features)
    # spin_up_mask = self.spin_up_mask.repeat((n_samples, 1, 1))
    # spin_down_mask = self.spin_down_mask.repeat((n_samples, 1, 1))

    # --- Single summations
    # up
    sum_spin_up = single_up_mask * single
    sum_spin_up = jnp.sum(sum_spin_up, axis=1, keepdims=True) / n_up
    #     sum_spin_up = jnp.repeat(sum_spin_up, n_electrons, axis=1)  # not needed in split

    # down
    sum_spin_down = single_down_mask * single
    sum_spin_down = jnp.sum(sum_spin_down, axis=1, keepdims=True) / n_down
    #     sum_spin_down = jnp.repeat(sum_spin_down, n_electrons, axis=1) # not needed in split

    # --- Pairwise summations
    sum_pairwise = jnp.repeat(jnp.expand_dims(pairwise, axis=1), n_electrons, axis=1)

    # up
    sum_pairwise_up = pairwise_up_mask * sum_pairwise
    sum_pairwise_up = jnp.sum(sum_pairwise_up, axis=2) / n_up

    # down
    sum_pairwise_down = pairwise_down_mask * sum_pairwise
    sum_pairwise_down = jnp.sum(sum_pairwise_down, axis=2) / n_down

    single = jnp.concatenate((single, sum_pairwise_up, sum_pairwise_down), axis=2)
    split = jnp.concatenate((sum_spin_up, sum_spin_down), axis=2)
    return single, split