
import jax.numpy as jnp
import numpy as np


def model(params, r_electrons, r_atoms, masks, n_up, n_down):
    n_samples, n_electrons = r_electrons.shape[:2]
    n_atoms = r_atoms.shape[1]

    # ae_vectors
    ae_vectors = compute_ae_vectors(r_electrons, r_atoms)

    # compute the inputs
    single, pairwise = compute_inputs(r_electrons, ae_vectors)
    # print(single.shape, pairwise.shape)

    # mix the inputs
    # print([x.shape for x in masks[0]])
    single = mixer(single, pairwise, n_electrons, n_up, n_down, *masks[0])

    # initial streams s0 and p0
    # print(single.shape, pairwise.shape, params['s0'].shape, params['p0'].shape)
    single = linear(params['s0'], single)
    pairwise = linear(params['p0'], pairwise)

    # intermediate layers including mix
    for (s_params, p_params), mask in zip(params['intermediate'], masks[1:]):
        # print(s_params.shape, p_params.shape, [x.shape for x in mask])

        single_mixed = mixer(single, pairwise, n_electrons, n_up, n_down, *mask)
        # print(single_mixed.shape)

        single = linear(s_params, single_mixed) + single
        pairwise = linear(p_params, pairwise) + pairwise
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
    n_down = n_electrons - n_up
    n_pairwise = n_electrons**2

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
        mask_up = mask_up.reshape(-1)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down.reshape(-1)

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


def compute_inputs(r_electrons, ae_vectors):
    n_samples, n_electrons, n_atoms = ae_vectors.shape[:3]

    ae_distances = jnp.linalg.norm(ae_vectors, axis=-1, keepdims=True)
    single_inputs = jnp.concatenate([ae_vectors, ae_distances], axis=-1)
    single_inputs = single_inputs.reshape(n_samples, n_electrons, 4 * n_atoms)

    re1 = jnp.expand_dims(r_electrons, axis=2)
    re2 = jnp.transpose(re1, [0, 2, 1, 3])

    ee_vectors = re1 - re2
    ee_distances = jnp.linalg.norm(ee_vectors, axis=-1, keepdims=True)
    pairwise_inputs = jnp.concatenate([ee_vectors, ee_distances], axis=-1)
    pairwise_inputs = pairwise_inputs.reshape(n_samples, n_electrons**2, 4)

    return single_inputs, pairwise_inputs


def linear(p: jnp.array,
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
    sum_spin_up = jnp.repeat(sum_spin_up, n_electrons, axis=1)  # not needed in split

    # down
    sum_spin_down = single_down_mask * single
    sum_spin_down = jnp.sum(sum_spin_down, axis=1, keepdims=True) / n_down
    sum_spin_down = jnp.repeat(sum_spin_down, n_electrons, axis=1) # not needed in split

    # --- Pairwise summations
    sum_pairwise = jnp.repeat(jnp.expand_dims(pairwise, axis=1), n_electrons, axis=1)

    # up
    sum_pairwise_up = pairwise_up_mask * sum_pairwise
    sum_pairwise_up = jnp.sum(sum_pairwise_up, axis=2) / n_up

    # down
    sum_pairwise_down = pairwise_down_mask * sum_pairwise
    sum_pairwise_down = jnp.sum(sum_pairwise_down, axis=2) / n_down

    features = jnp.concatenate((single, sum_pairwise_up, sum_pairwise_down, sum_spin_up, sum_spin_down), axis=2)
    # split = jnp.concatenate((sum_spin_up, sum_spin_down), axis=2)
    return features


if __name__ == '__main__':

    import jax.numpy as jnp
    import jax.random as rnd

    def count_mixed_features(n_sh, n_ph):
        #     n_sh_mix = 2 * n_ph + n_sh # change mixer
        return 3 * n_sh + 2 * n_ph


    def initialise_params(key,
                          n_atom: int,
                          n_up: int,
                          n_down: int,
                          n_layers: int = 2,
                          n_sh: int = 16,
                          n_ph: int = 8,
                          n_det: int = 1):
        '''


        Notes:
        zip(*([iter(nums)]*2) nice idiom for iterating over in sets of 2
        '''

        # count the number of input features
        n_sh_in = 4 * n_atom
        n_ph_in = 4

        # count the features in the intermediate layers
        n_sh_mix = count_mixed_features(n_sh, n_ph)

        params = {'envelopes': {}}

        # initial layers
        key, subkey = rnd.split(key)
        params['s0'] = rnd.normal(subkey, (count_mixed_features(n_sh_in, n_ph_in) + 1, n_sh))

        key, subkey = rnd.split(key)
        params['p0'] = rnd.normal(subkey, (n_ph_in + 1, n_ph))

        # intermediate layers
        key, *subkeys = rnd.split(key, num=(n_layers * 2))
        params['intermediate'] = [[rnd.normal(sk1, (n_sh_mix + 1, n_sh)), rnd.normal(sk2, (n_ph + 1, n_ph))]
                                  for sk1, sk2 in zip(*([iter(subkeys)] * 2))]

        # env_linear
        key, *subkeys = rnd.split(key, num=3)
        params['envelopes']['linear'] = [rnd.normal(subkeys[0], (n_det, n_up, n_sh + 1)),
                                         rnd.normal(subkeys[1], (n_det, n_down, n_sh + 1))]

        # env_sigma
        key, *subkeys = rnd.split(key, num=3)
        params['envelopes']['sigma'] = [rnd.normal(subkeys[0], (n_det, n_up, n_atom, 3, 3)),
                                        rnd.normal(subkeys[1], (n_det, n_down, n_atom, 3, 3))]

        # env_pi
        key, *subkeys = rnd.split(key, num=3)
        params['envelopes']['pi'] = [rnd.normal(subkeys[0], (n_det, n_up, n_atom)),
                                     rnd.normal(subkeys[1], (n_det, n_down, n_atom))]

        return params


    def create_atom_batch(r_atoms, n_samples):
        return jnp.repeat(jnp.expand_dims(r_atoms, axis=0), n_samples, axis=0)