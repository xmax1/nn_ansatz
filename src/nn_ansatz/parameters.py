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


def initialise_linear_layers(params, key, n_sh_split, n_sh_mix, n_down, n_sh, n_ph, n_sh_in, n_ph_in, n_layers, psplit_spins):

    # initial layers
    key, *subkeys = rnd.split(key, num=5)
    params['split0'] = init_linear(subkeys[0], (n_sh_in * (2 - int(n_down==0)), n_sh), bias=False)  # n_down==0 modifies the input when n_down is zero
    params['s0'] = init_linear(subkeys[1], (count_mixed_features(n_sh_in, n_ph_in, n_down), n_sh), bias=True)
    params['ps0'] = init_linear(subkeys[2], (n_ph_in, n_ph), bias=True)
    if psplit_spins: params['pd0'] = init_linear(subkeys[3], (n_ph_in, n_ph), bias=True)

    # intermediate layers
    key, *subkeys = rnd.split(key, num=((n_layers-1) * 4 + 1))

    for i, (sk1, sk2, sk3, sk4) in enumerate(zip(*([iter(subkeys)] * 4)), 1):  # idiom for iterating in sets of 2/3/...
        params['split%i' % i] = init_linear(sk1, (n_sh_split, n_sh), bias=False)
        params['s%i' % i] = init_linear(sk2, (n_sh_mix, n_sh), bias=True)
        params['ps%i' % i] = init_linear(sk3, (n_ph, n_ph), bias=True)  # final network layer doesn't have pairwise layer
        if psplit_spins: params['pd%i' % i] = init_linear(sk4, (n_ph, n_ph), bias=True)  # final network layer doesn't have pairwise layer

    i += 1
    key, *subkeys = rnd.split(key, num=5)
    params['split%i' % i] = init_linear(subkeys[0], (n_sh_split, n_sh//2), bias=False)
    params['s%i' % i] = init_linear(subkeys[1], (n_sh_mix, n_sh//2), bias=True)
    params['ps%i' % i] = init_linear(subkeys[2], (n_ph, n_ph//2), bias=True)  # final network layer doesn't have pairwise layer
    if psplit_spins: params['pd%i' % i] = init_linear(subkeys[3], (n_ph, n_ph//2), bias=True)  # final network layer doesn't have pairwise layer

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
    n_layers, n_sh, n_ph, n_det, n_in, n_sh_in, n_ph_in = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det, mol.n_in, mol.n_sh_in, mol.n_ph_in
    n_atoms, n_up, n_down = mol.n_atoms, mol.n_up, mol.n_down
    orbitals = mol.orbitals
    psplit_spins = mol.psplit_spins

    n_sh_mix = count_mixed_features(n_sh, n_ph, n_down)
    n_sh_split = n_sh * (2 - int(n_down==0)) # n_down==0 reduces the hidden dimension when n_down is zero
    
    params = OrderedDict()

    params, key = initialise_linear_layers(params, key, n_sh_split, n_sh_mix, n_down, n_sh, n_ph, n_sh_in, n_ph_in, n_layers, psplit_spins)

    if mol.backflow_coords:
        params['bf_up'] = init_linear(key, ((n_sh_split + n_sh_mix)//2, 3), bias=False)
        if not mol.n_down == 0: params['bf_down'] = init_linear(key, ((n_sh_split + n_sh_mix)//2, 3), bias=False)

    # env_linear
    for k in range(n_det):
        key, *subkeys = rnd.split(key, num=3)
        params['env_lin_up_k%i' % k] = init_linear(subkeys[0], ((n_sh_split + n_sh_mix)//2, n_up), bias=True)
        if not n_down == 0: params['env_lin_down_k%i' % k] = init_linear(subkeys[1], ((n_sh_split + n_sh_mix)//2, n_down), bias=True)

        
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
                params['env_sigma_up_m%i' % m] = params['env_sigma_up_m%i' % m].reshape(sigma_shape_up, order='F')  # so the 3 is in the right place
            #     params['env_sigma_up_m%i' % m] = init_einsum(k1, sigma_shape_up, bias=False)  # (3, 3 * n_det*n_spin)
                if not n_down == 0:
                    params['env_sigma_down_m%i' % m] = params['env_sigma_down_m%i' % m].reshape(sigma_shape_down, order='F') # so the 3 is in the right place
                    # params['env_sigma_down_m%i' % m] = init_einsum(k2, sigma_shape_down, bias=False)

            key, *subkeys = rnd.split(key, num=3)
            params['env_pi_up'] = jnp.stack([params['env_pi_up_%i' % i] for i in range(n_det*n_up)], axis=-1).reshape((n_atoms, n_det, n_up))
            params = {k: v for k, v in params.items() if 'env_pi_up_' not in k}
            # params['env_pi_up'] = jnp.squeeze(init_einsum(subkeys[0], (n_atoms, 1, n_det, n_up)), axis=1)
            
            if not n_down == 0:
                params['env_pi_down'] = jnp.stack([params['env_pi_down_%i' % i] for i in range(n_det*n_down)], axis=-1).reshape((n_atoms, n_det, n_down))    
                params = {k: v for k, v in params.items() if 'env_pi_down_' not in k}
            #     params['env_pi_down'] = jnp.squeeze(init_einsum(subkeys[1], (n_atoms, 1, n_det, n_down)), axis=1)
            
            
    
    # if mol.jastrow:
    #     if mol.backflow_coords:
    #         params['jf_bf'] = init_linear(key, (5, 1), bias=False)


            # # env_pi

    # values, tree_map = tree_flatten(params)  # get the tree_map and then flatten
    # values = [v * INIT for v in values]  # scale all of the parameters
    # params = tree_unflatten(tree_map, values)  # put the tree back together with the map
    return params


def initialise_linear_layers_d0s(d0s, n_up, n_down, n_sh, n_ph, n_layers, psplit_spins):
    n_el = n_up + n_down
    if psplit_spins:
        n_same = n_up**2 + n_down**2
        n_diff = 2 * n_up * n_down
    else:
        n_same = n_el**2
        n_diff = 0

    # initial layers
    d0s['split0'] = jnp.zeros((1, n_sh))
    d0s['s0'] = jnp.zeros((n_el, n_sh))
    d0s['ps0'] = jnp.zeros((n_same, n_ph))
    if psplit_spins: d0s['pd0'] = jnp.zeros((n_diff, n_ph))

    for i in range(1, n_layers):
        d0s['split%i' % i] = jnp.zeros((1, n_sh))
        d0s['s%i' % i] = jnp.zeros((n_el, n_sh))
        d0s['ps%i' % i] = jnp.zeros((n_same, n_ph))  # final network layer doesn't have pairwise layer
        if psplit_spins: d0s['pd%i' % i] = jnp.zeros((n_diff, n_ph))  # final network layer doesn't have pairwise layer

    i+=1
    d0s['split%i' % i] = jnp.zeros((1, n_sh//2))
    d0s['s%i' % i] = jnp.zeros((n_el, n_sh//2))
    d0s['ps%i' % i] = jnp.zeros((n_same, n_ph//2))  # final network layer doesn't have pairwise layer
    if psplit_spins: d0s['pd%i' % i] = jnp.zeros((n_diff, n_ph//2))  # final network layer doesn't have pairwise layer

    return d0s


def initialise_d0s(mol, expand=False):
    n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
    n_el, n_pairwise, n_atoms, n_up, n_down = mol.n_el, mol.n_pairwise, mol.n_atoms, mol.n_up, mol.n_down

    d0s = OrderedDict()

    d0s = initialise_linear_layers_d0s(d0s, n_up, n_down, n_sh, n_ph, n_layers, mol.psplit_spins)

    if mol.backflow_coords:
        d0s['bf_up'] = jnp.zeros((n_up, 3))
        if not mol.n_down == 0: d0s['bf_down'] = jnp.zeros((n_down, 3))

    for k in range(n_det):
        d0s['env_lin_up_k%i' % k] = jnp.zeros((n_up, n_up))
        if not n_down == 0: d0s['env_lin_down_k%i' % k] = jnp.zeros((n_down, n_down))

    if not mol.einsum:
        
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

        if not mol.system == 'HEG':
            n_exponent_dim = 3 if mol.orbitals == 'anisotropic' else 1

            for m in range(n_atoms):
                d0s['env_sigma_up_m%i' % m] = jnp.zeros((n_up, n_exponent_dim, n_det, n_up))
                if not n_down == 0: d0s['env_sigma_down_m%i' % m] = jnp.zeros((n_down, n_exponent_dim, n_det, n_down))
            
            d0s['env_pi_up'] = jnp.zeros((n_det, n_up, n_up))
            
            if not n_down == 0:
                d0s['env_pi_down'] = jnp.zeros((n_det, n_down, n_down))

    # if mol.jastrow:
    #     if mol.backflow_coords:
    #         d0s['jf_bf'] = jnp.zeros((n_el**2, 1))

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