import jax.numpy as jnp
import jax.random as rnd
from jax.tree_util import tree_flatten, tree_unflatten
from jax.nn.initializers import orthogonal

from collections import OrderedDict
import os

INIT = 0.01

init_orthogonal = orthogonal()


def init_linear(key, shape, bias=True, bias_axis=0):
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


def initialise_params(mol, key):
    if len(key.shape) > 1:
        key = key[0]
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
    params['p0'] = init_linear(subkeys[2], (n_ph_in, n_ph), bias=True)

    # intermediate layers
    key, *subkeys = rnd.split(key, num=(n_layers * 3 + 1))
    params['intermediate'] = [[init_linear(sk2, (n_sh_split, n_sh), bias=False),
                               init_linear(sk1, (n_sh_mix, n_sh), bias=True),
                               init_linear(sk3, (n_ph, n_ph), bias=True)]
                              for sk1, sk2, sk3 in zip(*([iter(subkeys)] * 3))]

    # env_linear
    params['envelopes'] = OrderedDict()

    key, *subkeys = rnd.split(key, num=3)
    params['envelopes']['linear'] = [init_linear(subkeys[0], (n_sh, n_det * n_up), bias=True),
                                     init_linear(subkeys[1], (n_sh, n_det * n_down), bias=True)]
    # params['envelopes']['linear'] = [init_linear(subkeys[0], (n_det, n_up, n_sh), bias=True),
    #                                  init_linear(subkeys[1], (n_det, n_down, n_sh), bias=True)]

    # env_sigma
    key, *subkeys = rnd.split(key, num=3)
    # SIGMA BROADCAST
    params['envelopes']['sigma'] = OrderedDict()
    params['envelopes']['sigma']['up'] = init_linear(subkeys[0], (n_det, n_up, n_atoms, 3, 3), bias=False)
    params['envelopes']['sigma']['down'] = init_linear(subkeys[1], (n_det, n_down, n_atoms, 3, 3), bias=False)
    # SIGMA LOOPY list(atom1, atom2)... atom1 = list( (3x3) n_det x n_spins)
    # params['envelopes']['sigma'] = OrderedDict()
    # params['envelopes']['sigma']['up'] = [[init_linear_layer(subkeys[0], (3, 3), False) for _ in range(n_det * n_up)] for _ in range(n_atoms)]
    # params['envelopes']['sigma']['down'] = [[init_linear_layer(subkeys[0], (3, 3), False) for _ in range(n_det * n_down)] for _ in range(n_atoms)]


    # env_pi
    key, *subkeys = rnd.split(key, num=3)
    up_shape, down_shape = (n_det * n_up * n_atoms,), (n_det * n_down * n_atoms,)
    x = jnp.ones(up_shape) + rnd.normal(subkeys[0], up_shape) * 0.01
    y = jnp.ones(down_shape) + rnd.normal(subkeys[1], down_shape) * 0.01

    x = [x[:, None] for x in jnp.split(x, n_det * n_up)]
    y = [y[:, None] for y in jnp.split(y, n_det * n_down)]
    params['envelopes']['pi'] = [x, y]

    # values, tree_map = tree_flatten(params)  # get the tree_map and then flatten
    # values = [v * INIT for v in values]  # scale all of the parameters
    # params = tree_unflatten(tree_map, values)  # put the tree back together with the map
    return params


def initialise_d0s(mol, expand=False):
    n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
    n_el, n_pairwise, n_atoms, n_up, n_down = mol.n_el, mol.n_pairwise, mol.n_atoms, mol.n_up, mol.n_down

    d0s = OrderedDict()

    # initial layers
    d0s['split0'] = jnp.zeros((1, n_sh))
    d0s['s0'] = jnp.zeros((n_el, n_sh))
    d0s['p0'] = jnp.zeros((n_pairwise, n_ph))

    # intermediate layers
    d0s['intermediate'] = [[jnp.zeros((1, n_sh)),
                               jnp.zeros((n_el, n_sh)),
                               jnp.zeros((n_pairwise, n_ph))]
                               for _ in range(n_layers)]

    d0s['envelopes'] = OrderedDict()
    d0s['envelopes']['linear'] = [jnp.zeros((n_up, n_det * n_up)),
                                     jnp.zeros((n_down, n_det * n_down))]

    # SIGMA BROADCAST
    d0s['envelopes']['sigma'] = OrderedDict()
    d0s['envelopes']['sigma']['up'] = [jnp.zeros((n_up, 3 * n_det * n_up)) for _ in range(n_atoms)]
    d0s['envelopes']['sigma']['down'] = [jnp.zeros((n_down, 3 * n_det * n_down)) for _ in range(n_atoms)]

    # SIGMA LOOPY
    # d0s['envelopes']['sigma'] = OrderedDict()
    # d0s['envelopes']['sigma']['up'] = [[jnp.zeros((n_up, 3)) for _ in range(n_det * n_up)] for _ in range(n_atoms)]
    # d0s['envelopes']['sigma']['down'] = [[jnp.zeros((n_down, 3)) for _ in range(n_det * n_down)] for _ in range(n_atoms)]

    # d0s['envelopes']['pi'] = [[jnp.squeeze(x, axis=-1) for x in jnp.split(jnp.zeros((n_up, n_det * n_up)), n_det*n_up, axis=1)],
    #                              [jnp.squeeze(x, axis=-1) for x in jnp.split(jnp.zeros((n_down, n_det * n_down)), n_det*n_down, axis=1)]]
    d0s['envelopes']['pi'] = [jnp.split(jnp.zeros((n_up, n_det * n_up)), n_det * n_up, axis=1),
                                 jnp.split(jnp.zeros((n_down, n_det * n_down)), n_det * n_down, axis=1)]

    if expand: # distinguish between the cases 1- used to create a partial function (don't expand) 2- used to find the sensitivities (expand)
        d0s = expand_d0s(d0s, mol.n_devices, mol.n_walkers_per_device)

    return d0s


def expand_d0s(d0s, n_devices, n_walkers_per_device):
    d0s, tree_map = tree_flatten(d0s) 
    d0s = [jnp.repeat(jnp.expand_dims(d0, axis=0), n_walkers_per_device, axis=0) for d0 in d0s]
    if bool(os.environ.get('DISTRIBUTE')) is True:
        d0s = [jnp.repeat(jnp.expand_dims(d0, axis=0), n_devices, axis=0) for d0 in d0s]
    d0s = tree_unflatten(tree_map, d0s)
    return d0s