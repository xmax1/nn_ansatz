import jax.numpy as jnp
import jax.random as rnd
from jax.tree_util import tree_flatten, tree_unflatten
from jax.nn.initializers import orthogonal

from collections import OrderedDict

INIT = 0.01

init_orthogonal = orthogonal()


def init_linear(key, shape, bias=True):
    if len(shape) == 5:
        n_det, n_spin, n_atom, _, _ = shape
        subkeys = rnd.split(key, num=jnp.prod(n_det * n_spin))
        new_shape = (3, 3)
        # p = jnp.concatenate([init_linear_layer(k, new_shape, bias)[None, ...] for k in subkeys], axis=0)
        # p = p.reshape(shape)

        p = [jnp.concatenate([init_linear_layer(k, new_shape, bias) for k in subkeys], axis=-1)
             for _ in range(n_atom)]

        # p = jnp.concatenate([init_linear_layer(k, new_shape, bias)[..., None] for k in subkeys], axis=-1)
        # p = p.reshape(3, n_atom, -1)
        # p = [jnp.squeeze(x) for x in jnp.split(p, n_atom, axis=1)]

        # subkeys = rnd.split(key, num=n_atom)
        # p = [init_linear_layer(k, (3, n_det * n_spin * 3), bias) for k in subkeys]

    elif len(shape) == 3:
        subkeys = rnd.split(key, num=shape[0])
        new_shape = shape[1:]
        p = jnp.concatenate([init_linear_layer(k, new_shape, bias, bias_axis=-1)[None, ...] for k in subkeys], axis=0)

    else:
        p = init_linear_layer(key, shape, bias)

    return p


def init_linear_layer(key, shape, bias, bias_axis=0):
    key, subkey = rnd.split(key)
    p = init_orthogonal(key, shape)
    if bias:
        shape = list(shape)
        shape[bias_axis] = 1
        b = rnd.normal(subkey, tuple(shape))
        p = jnp.concatenate([p, b], axis=bias_axis)
    return p


def count_mixed_features(n_sh, n_ph):
    #     n_sh_mix = 2 * n_ph + n_sh # change mixer
    return n_sh + 2 * n_ph


def initialise_params(key,
                      mol):
    n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
    n_atoms, n_up, n_down = mol.n_atoms, mol.n_up, mol.n_down
    '''


    Notes:
    zip(*([iter(nums)]*2) nice idiom for iterating over in sets of 2
    '''
    # count the number of input features
    n_sh_in = 4 * n_atoms
    n_ph_in = 4

    # count the features in the intermediate layers
    n_sh_mix = count_mixed_features(n_sh, n_ph)
    n_sh_split = 2 * n_sh

    params = OrderedDict()

    # initial layers
    key, *subkeys = rnd.split(key, num=4)
    params['split0'] = init_linear(subkeys[0], (n_sh_in * 2, n_sh), bias=False)
    params['s0'] = init_linear(subkeys[1], (count_mixed_features(n_sh_in, n_ph_in), n_sh), bias=True)
    params['p0'] = init_linear(subkeys[2], (n_ph_in , n_ph), bias=True)

    # intermediate layers
    key, *subkeys = rnd.split(key, num=(n_layers * 3 + 1))
    params['intermediate'] = [[init_linear(sk2, (n_sh_split, n_sh), bias=False),
                               init_linear(sk1, (n_sh_mix, n_sh), bias=True),
                               init_linear(sk3, (n_ph, n_ph), bias=True)]
                              for sk1, sk2, sk3 in zip(*([iter(subkeys)] * 3))]

    # env_linear
    params['envelopes'] = OrderedDict()

    key, *subkeys = rnd.split(key, num=3)
    params['envelopes']['linear'] = [init_linear(subkeys[0], (n_det, n_up, n_sh), bias=True),
                                     init_linear(subkeys[1], (n_det, n_down, n_sh), bias=True)]

    # env_sigma
    key, *subkeys = rnd.split(key, num=3)
    params['envelopes']['sigma'] = [init_linear(subkeys[0], (n_det, n_up, n_atoms, 3, 3), bias=False),
                                    init_linear(subkeys[1], (n_det, n_down, n_atoms, 3, 3), bias=False)]

    # env_pi
    key, *subkeys = rnd.split(key, num=3)
    up_shape, down_shape = (1, n_det * n_up * n_atoms), (1, n_det * n_down * n_atoms)
    params['envelopes']['pi'] = [jnp.ones(up_shape) + rnd.normal(subkeys[0], up_shape) * 0.01,
                                 jnp.ones(down_shape) + rnd.normal(subkeys[1], down_shape) * 0.01]

    # values, tree_map = tree_flatten(params)  # get the tree_map and then flatten
    # values = [v * INIT for v in values]  # scale all of the parameters
    # params = tree_unflatten(tree_map, values)  # put the tree back together with the map
    return params


def initialise_params_dep(key,
                      mol):
    n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
    n_atoms, n_up, n_down = mol.n_atoms, mol.n_up, mol.n_down
    '''


    Notes:
    zip(*([iter(nums)]*2) nice idiom for iterating over in sets of 2
    '''
    # count the number of input features
    n_sh_in = 4 * n_atoms
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
    params['envelopes']['sigma'] = [rnd.normal(subkeys[0], (n_det, n_up, n_atoms, 3, 3)),
                                    rnd.normal(subkeys[1], (n_det, n_down, n_atoms, 3, 3))]

    # env_pi
    key, *subkeys = rnd.split(key, num=3)
    params['envelopes']['pi'] = [rnd.normal(subkeys[0], (n_det, n_up, n_atoms)),
                                 rnd.normal(subkeys[1], (n_det, n_down, n_atoms))]

    values, tree_map = tree_flatten(params)  # get the tree_map and then flatten
    values = [v * INIT for v in values]  # scale all of the parameters
    params = tree_unflatten(tree_map, values)  # put the tree back together with the map
    return params