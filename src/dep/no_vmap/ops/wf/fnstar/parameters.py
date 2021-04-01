import jax.numpy as jnp
import jax.random as rnd
from collections import OrderedDict


def count_mixed_features(n_sh, n_ph):
    #     n_sh_mix = 2 * n_ph + n_sh # change mixer
    return n_sh + 2 * n_ph


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
    n_sh_split = 2 * n_sh

    params = OrderedDict()

    # initial layers
    key, *subkeys = rnd.split(key, num=4)
    params['split0'] = rnd.normal(subkeys[2], (n_sh_in * 2, n_sh))
    params['s0'] = rnd.normal(subkeys[0], (count_mixed_features(n_sh_in, n_ph_in) + 1, n_sh))
    params['p0'] = rnd.normal(subkeys[1], (n_ph_in + 1, n_ph))

    # intermediate layers
    key, *subkeys = rnd.split(key, num=(n_layers * 3 + 1))
    params['intermediate'] = [[rnd.normal(sk1, (n_sh_mix + 1, n_sh)),
                               rnd.normal(sk2, (n_sh_split, n_sh)),
                               rnd.normal(sk3, (n_ph + 1, n_ph))]
                              for sk1, sk2, sk3 in zip(*([iter(subkeys)] * 3))]

    # env_linear
    params['envelopes'] = OrderedDict()

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