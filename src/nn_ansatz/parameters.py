import jax.numpy as jnp
import jax.random as rnd
from jax.tree_util import tree_flatten, tree_unflatten
from jax.nn.initializers import orthogonal

from collections import OrderedDict
import os

INIT = 0.01

init_orthogonal = orthogonal()


def init_linear(key, shape, bias, bias_axis=0):
    key, subkey = rnd.split(key)
    p = init_orthogonal(key, shape)
    if bias:
        shape = list(shape)
        shape[bias_axis] = 1
        b = rnd.normal(subkey, tuple(shape))
        p = jnp.concatenate([p, b], axis=bias_axis)
    return p


def init_sigma(key, shape, bias=True, bias_axis=0):
    if len(shape) == 3:  # anisotropic
        new_shape = (3, 3)
        p = jnp.concatenate([init_linear(k, new_shape, bias=False) for k in rnd.split(key, num=shape[-1])], axis=-1)
    if len(shape) == 2:  # isotropic
        p = unit_plus_noise(shape, key)
    return p


def unit_plus_noise(shape, key):
    return jnp.ones(shape) + rnd.normal(key, shape) * 0.01



def count_mixed_features(n_sh, n_ph, n_down):
    #     n_sh_mix = 2 * n_ph + n_sh # change mixer
    return n_sh + n_ph * (2 - int(n_down==0))


def initialise_linear_layers(params, key, n_in, n_atoms, n_down, n_sh, n_ph, n_layers):
    # count the number of input features
    n_sh_in = n_in * n_atoms
    n_ph_in = n_in

    # count the features in the intermediate layers
    n_sh_mix = count_mixed_features(n_sh, n_ph, n_down)
    n_sh_split = n_sh * (2 - int(n_down==0)) # n_down==0 reduces the hidden dimension when n_down is zero

    # initial layers
    key, *subkeys = rnd.split(key, num=4)
    params['split0'] = init_linear(subkeys[0], (n_sh_in * (2 - int(n_down==0)), n_sh), bias=False)  # n_down==0 modifies the input when n_down is zero
    params['s0'] = init_linear(subkeys[1], (count_mixed_features(n_sh_in, n_ph_in, n_down), n_sh), bias=True)
    params['p0'] = init_linear(subkeys[2], (n_ph_in, n_ph), bias=True)

    # intermediate layers
    key, *subkeys = rnd.split(key, num=(n_layers * 3 + 1))

    for i, (sk1, sk2, sk3) in enumerate(zip(*([iter(subkeys)] * 3)), 1):  # idiom for iterating in sets of 2/3/...
        params['split%i' % i] = init_linear(sk1, (n_sh_split, n_sh), bias=False)
        params['s%i' % i] = init_linear(sk2, (n_sh_mix, n_sh), bias=True)
        params['p%i' % i] = init_linear(sk3, (n_ph, n_ph), bias=True)

    return params, key


def init_einsum(key, shape, bias=False):
    if len(shape) > 2:
        n_layers = jnp.prod(jnp.array(shape[2:]))
        layers = []
        subkeys = rnd.split(key, num=n_layers)
        for n_layer, sk in zip(range(n_layers), subkeys):
            layer = init_linear(sk, shape[:2], bias)
            layers.append(layer)
        layers = jnp.stack(layers, axis=-1)
        return layers.reshape(shape)
    else:
        return init_linear(key, shape)
        


def initialise_params(mol, key):
    if len(key.shape) > 1:
        key = key[0]
    n_layers, n_sh, n_ph, n_det, n_in = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det, mol.n_in
    n_atoms, n_up, n_down = mol.n_atoms, mol.n_up, mol.n_down
    orbitals = mol.orbitals
    
    params = OrderedDict()

    params, key = initialise_linear_layers(params, key, n_in, n_atoms, n_down, n_sh, n_ph, n_layers)

    # env_linear
    key, *subkeys = rnd.split(key, num=3)
    params['env_lin_up'] = init_linear(subkeys[0], (n_sh, n_det * n_up), bias=True)
    if not n_down == 0: params['env_lin_down'] = init_linear(subkeys[1], (n_sh, n_det * n_down), bias=True)

    # if not mol.einsum:
        
    # env_sigma
    if not mol.orbitals == 'real_plane_waves':
        key, *subkeys = rnd.split(key, num=3)
        sigma_shape_up = (3, 3, n_det * n_up) if orbitals == 'anisotropic' else (1, n_det * n_up)
        sigma_shape_down = (3, 3, n_det * n_down) if orbitals == 'anisotropic' else (1, n_det * n_down)
        
        for m, (k1, k2) in enumerate(zip(rnd.split(subkeys[0], num=n_atoms), rnd.split(subkeys[1], num=n_atoms))):
            params['env_sigma_up_m%i' % m] = init_sigma(k1, sigma_shape_up, bias=False)  # (3, 3 * n_det*n_spin)
            if not n_down == 0:
                params['env_sigma_down_m%i' % m] = init_sigma(k2, sigma_shape_down, bias=False)

        # env_pi
        key, *subkeys = rnd.split(key, num=3)
        for i, k in enumerate(rnd.split(subkeys[0], num=n_up*n_det)):
            params['env_pi_up_%i' % i] = unit_plus_noise((n_atoms, 1), k)
        
        if not n_down == 0:
            for i, k in enumerate(rnd.split(subkeys[1], num=n_down*n_det)):
                params['env_pi_down_%i' % i] = unit_plus_noise((n_atoms, 1), k)
    if mol.einsum:
        # env_sigma
        if not mol.orbitals == 'real_plane_waves':
            key, *subkeys = rnd.split(key, num=3)
            sigma_shape_up = (3, 3, n_det, n_up) if orbitals == 'anisotropic' else (1, n_det, n_up)
            sigma_shape_down = (3, 3, n_det, n_down) if orbitals == 'anisotropic' else (1, n_det, n_down)
            
            for m, (k1, k2) in enumerate(zip(rnd.split(subkeys[0], num=n_atoms), rnd.split(subkeys[1], num=n_atoms))):
                params['env_sigma_up_m%i' % m] = params['env_sigma_up_m%i' % m].reshape(sigma_shape_up, order='F')
            #     params['env_sigma_up_m%i' % m] = init_einsum(k1, sigma_shape_up, bias=False)  # (3, 3 * n_det*n_spin)
                if not n_down == 0:
                    params['env_sigma_down_m%i' % m] = params['env_sigma_down_m%i' % m].reshape(sigma_shape_down, order='F')
                    # params['env_sigma_down_m%i' % m] = init_einsum(k2, sigma_shape_down, bias=False)

            key, *subkeys = rnd.split(key, num=3)
            params['env_pi_up'] = jnp.stack([params['env_pi_up_%i' % i] for i in range(n_det*n_up)], axis=-1).reshape((n_atoms, n_det, n_up))
            params = {k: v for k, v in params.items() if 'env_pi_up_' not in k}
            # params['env_pi_up'] = jnp.squeeze(init_einsum(subkeys[0], (n_atoms, 1, n_det, n_up)), axis=1)
            
            if not n_down == 0:
                params['env_pi_down'] = jnp.stack([params['env_pi_down_%i' % i] for i in range(n_det*n_down)], axis=-1).reshape((n_atoms, n_det, n_down))    
                params = {k: v for k, v in params.items() if 'env_pi_down_' not in k}
            #     params['env_pi_down'] = jnp.squeeze(init_einsum(subkeys[1], (n_atoms, 1, n_det, n_down)), axis=1)
            
            
            


            # # env_pi

    # values, tree_map = tree_flatten(params)  # get the tree_map and then flatten
    # values = [v * INIT for v in values]  # scale all of the parameters
    # params = tree_unflatten(tree_map, values)  # put the tree back together with the map
    return params


def initialise_linear_layers_d0s(d0s, n_el, n_pairwise, n_sh, n_ph, n_layers):
    # initial layers
    d0s['split0'] = jnp.zeros((1, n_sh))
    d0s['s0'] = jnp.zeros((n_el, n_sh))
    d0s['p0'] = jnp.zeros((n_pairwise, n_ph))

    for i in range(1, n_layers+1):
        d0s['split%i' % i] = jnp.zeros((1, n_sh))
        d0s['s%i' % i] = jnp.zeros((n_el, n_sh))
        d0s['p%i' % i] = jnp.zeros((n_pairwise, n_ph))

    return d0s


def initialise_d0s(mol, expand=False):
    n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
    n_el, n_pairwise, n_atoms, n_up, n_down = mol.n_el, mol.n_pairwise, mol.n_atoms, mol.n_up, mol.n_down

    d0s = OrderedDict()

    d0s = initialise_linear_layers_d0s(d0s, n_el, n_pairwise, n_sh, n_ph, n_layers)

    d0s['env_lin_up'] = jnp.zeros((n_up, n_det * n_up))
    if not n_down == 0: d0s['env_lin_down'] = jnp.zeros((n_down, n_det * n_down))

    if not mol.einsum:
        
        # SIGMA BROADCAST
        if not mol.system == 'HEG':
            n_exponent_dim = 3 if mol.orbitals == 'anisotropic' else 1

            for m in range(n_atoms):
                d0s['env_sigma_up_m%i' % m] = jnp.zeros((n_up, n_exponent_dim * n_det * n_up))
                if not n_down == 0: d0s['env_sigma_down_m%i' % m] = jnp.zeros((n_down, n_exponent_dim * n_det * n_down))
            
            for i in range(n_det * n_up):
                d0s['env_pi_up_%i' % i] = jnp.zeros((n_up, 1))
            
            if not n_down == 0:
                for i in range(n_det * n_down):
                    d0s['env_pi_down_%i' % i] = jnp.zeros((n_down, 1))

    else:

        # SIGMA BROADCAST
        if not mol.system == 'HEG':
            n_exponent_dim = 3 if mol.orbitals == 'anisotropic' else 1

            for m in range(n_atoms):
                d0s['env_sigma_up_m%i' % m] = jnp.zeros((n_up, n_exponent_dim, n_det, n_up))
                if not n_down == 0: d0s['env_sigma_down_m%i' % m] = jnp.zeros((n_down, n_exponent_dim, n_det, n_down))
            
            d0s['env_pi_up'] = jnp.zeros((n_det,n_up, n_up))
            
            if not n_down == 0:
                d0s['env_pi_down'] = jnp.zeros((n_det,n_down, n_down))

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