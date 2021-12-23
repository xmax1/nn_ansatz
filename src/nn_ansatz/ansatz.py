from typing import Callable
from functools import partial, reduce

import jax.numpy as jnp
from jax import vmap


from .parameters import initialise_d0s
from .ansatz_base import *


def keep_in_boundary(walkers, basis, inv_basis):
    talkers = transform_vector_space(walkers, inv_basis)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0., talkers + 1., talkers)
    talkers = transform_vector_space(talkers, basis)
    return talkers


def create_wf(mol, kfac: bool=False, orbitals: bool=False, signed: bool=False, distribute=False):
    ''' initializes the wave function ansatz for various applications '''

    print('creating wf')
    
    masks = create_masks(mol.n_el, mol.n_up, mol.n_layers)

    _compute_single_stream_vectors = partial(compute_single_stream_vectors_i, r_atoms=mol.r_atoms, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis)
    _compute_inputs = partial(compute_inputs_i, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis, input_activation_nonlinearity=mol.input_activation_nonlinearity, kpoints=mol.kpoints)
    _compute_orbitals, _sum_orbitals = create_orbitals(orbitals=mol.orbitals, n_el=mol.n_el, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis, einsum=mol.einsum, kpoints=mol.kpoints)
    _compute_jastrow = create_jastrow_factor(mol.n_el, mol.n_up, mol.volume, mol.density_parameter, mol.basis, mol.inv_basis) if mol.jastrow else None
    _backflow_block = partial(backflow_block,
                              masks=masks,
                              n_up=mol.n_up,
                              n_down=mol.n_down,
                              n_el=mol.n_el,
                              backflow_coords=mol.backflow_coords,
                    
                              _compute_single_stream_vectors=_compute_single_stream_vectors,
                              _compute_inputs=_compute_inputs,
                              _linear= partial(linear, nonlinearity=mol.nonlinearity),
                              _linear_pairwise=partial(linear_pairwise, nonlinearity=mol.nonlinearity, psplit_spins=mol.psplit_spins))

    _wf_orbitals = partial(wf_orbitals, 
                           n_el=mol.n_el,
                           n_up=mol.n_up,
                           n_down=mol.n_down,
                           n_det=mol.n_det,
                           _compute_orbitals=_compute_orbitals,
                           _sum_orbitals=_sum_orbitals)

    def _wf(params, walkers, d0s):
        
        activations = []

        if len(walkers.shape) == 1:  # this is a hack to get around the jvp
            walkers = walkers.reshape(mol.n_el, 3)

        if mol.inv_basis is not None:
            walkers = transform_vector_space(walkers, mol.inv_basis, on=True)

        data_up, data_down, single_stream_vectors = _backflow_block(params, walkers, activations, d0s)
        
        orb_up, orb_down, activations = _wf_orbitals(params, data_up, data_down, single_stream_vectors, activations, d0s)
        
        log_psi, sign = logabssumdet(orb_up, orb_down)

        if _compute_jastrow is not None:
    
            jastrow_factor = _compute_jastrow(params, walkers, activations, d0s)
            sign *= jnp.sign(jastrow_factor)
            log_psi += jastrow_factor
        
        if kfac:
            return log_psi, activations
        elif signed:
            return log_psi, sign
        else:
            return log_psi
    
    d0s = initialise_d0s(mol)

    if orbitals:
        def _orbs(params, walkers):
            activations = []

            if len(walkers.shape) == 1:  # this is a hack to get around the jvp
                walkers = walkers.reshape(mol.n_el, 3)

            if mol.inv_basis is not None:
                walkers = transform_vector_space(walkers, mol.inv_basis, on=True)

            data_up, data_down, single_stream_vectors = _backflow_block(params, walkers, activations, d0s)
        
            orb_up, orb_down, _ = _wf_orbitals(params, data_up, data_down, single_stream_vectors, activations, d0s)
            return orb_up, orb_down
        return vmap(_orbs, in_axes=(None, 0))

    if kfac:
        return vmap(_wf, in_axes=(None, 0, 0))

    _partial_wf = partial(_wf, d0s=d0s)
    _vwf = vmap(_partial_wf, in_axes=(None, 0))
    
    return _vwf


def backflow_block(params: dict, 
                   walkers: jnp.array, 
                   activations: list,
                   d0s: dict, 
                   masks: list,
                   n_up: int,
                   n_down: int,
                   n_el: int,
                   backflow_coords: bool,
                
                   _compute_single_stream_vectors: Callable,
                   _compute_inputs: Callable,
                   _linear: Callable = partial(linear, nonlinearity='tanh'),
                   _linear_pairwise: Callable = partial(linear_pairwise, nonlinearity='tanh')):

    single_stream_vectors = _compute_single_stream_vectors(walkers)  # (n_el, n_atom, 3)
    single, pairwise = _compute_inputs(walkers, single_stream_vectors)  # (n_el, 3), (n_pairwise, 4)
    single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *masks[0])

    for i, mask in enumerate(masks[1:], 0):
    
        split = linear_split(params['split%i'%i], split, activations, d0s['split%i'%i])
        single = _linear(params['s%i'%i], single_mixed, split, activations, d0s['s%i'%i], single)
        pairwise = _linear_pairwise(params, 
                                    i,
                                    pairwise, 
                                    activations, 
                                    d0s,
                                    pairwise,
                                    n_up, 
                                    n_down)

        single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *mask)

    single = jnp.concatenate([single_mixed, split.repeat(n_el, axis=0)], axis=-1)
    data_up, data_down = jnp.split(single, [n_up], axis=0)

    if backflow_coords:
        new_coords = linear_split(params['bf_up'], data_up, activations, d0s['bf_up'])[:, None, :]
        if not n_down == 0:
            new_coords_down = linear_split(params['bf_down'], data_down, activations, d0s['bf_down'])[:, None, :]
            new_coords = layer_activation(jnp.concatenate([new_coords, new_coords_down], axis=0), nonlinearity='tanh')
        single_stream_vectors += new_coords
    return data_up, data_down, single_stream_vectors


def wf_orbitals(params: dict, 
                data_up: jnp.array, 
                data_down: jnp.array,
                single_stream_vectors: jnp.array,
                activations: list,
                d0s: dict, 

                n_up: int,
                n_down: int,
                n_el: int,
                n_det: int,
            
                _compute_orbitals: Callable,
                _sum_orbitals: Callable):

    factors_up, factors_down = [], []
    for k in range(n_det):
        factor_up = env_linear_i(params['env_lin_up_k%i' % k], data_up, activations, d0s['env_lin_up_k%i' % k])
        factors_up.append(factor_up)
        if not n_up == n_el: 
            factor_down = env_linear_i(params['env_lin_down_k%i' % k], data_down, activations, d0s['env_lin_down_k%i' % k])
            factors_down.append(factor_down)

    factor_up = jnp.stack(factors_up, axis=0)
    factor_down = jnp.stack(factors_down, axis=0) if not n_up == n_el else None
    single_stream_vectors = split_and_squeeze(single_stream_vectors, axis=1)  # [n_atom * (n_el, 3)]

    exp_up = []
    exp_down = []
    for m, single_stream_vector in enumerate(single_stream_vectors):
        ss_up_m, ss_down_m = jnp.split(single_stream_vector, [n_up], axis=0)  # (n_up, 3), (n_down, 3)
        exp_up_m = _compute_orbitals(params.get('env_sigma_up_m%i' % m), ss_up_m, d0s.get('env_sigma_up_m%i' % m), activations) 
        exp_up.append(exp_up_m)
        
        if not n_up == n_el:
            exp_down_m = _compute_orbitals(params.get('env_sigma_down_m%i' % m), ss_down_m, d0s.get('env_sigma_down_m%i' % m), activations)
            exp_down.append(exp_down_m)
    
    exp_up = jnp.stack(exp_up, axis=-1)
    orb_up = _sum_orbitals(params, factor_up, exp_up, activations, 'up', d0s)
    
    orb_down = None
    if not n_up == n_el:
        exp_down = jnp.stack(exp_down, axis=-1)
        orb_down = _sum_orbitals(params, factor_down, exp_down, activations, 'down', d0s)

    return orb_up, orb_down, activations


def create_orbitals(orbitals='anisotropic',
                    n_el: Optional[int]=None,
                    basis: Optional[jnp.array]=None,
                    inv_basis: Optional[jnp.array]=None,
                    pbc: bool=False,
                    einsum: bool=False,
                    kpoints: Optional[jnp.array] = None):

    if orbitals == 'anisotropic':
        _compute_orbitals = partial(anisotropic_orbitals, basis=basis, inv_basis=inv_basis, pbc=pbc, einsum=einsum)
        _sum_orbitals = partial(env_pi_i, einsum=einsum)

    if 'isotropic' in orbitals:
        _compute_orbitals = partial(isotropic_orbitals, pbc=pbc, basis=basis, inv_basis=inv_basis, orbitals=orbitals, einsum=einsum)
        _sum_orbitals = partial(env_pi_i, einsum=einsum)
    
    if orbitals == 'real_plane_waves':
        _compute_orbitals = partial(real_plane_wave_orbitals, kpoints=kpoints)
        def _sum_orbitals(params: jnp.array,
                          factor: jnp.array,
                          orbital: jnp.array,
                          activations: list,
                          spin: str,
                          d0s):
            # orbital (n_spin_i, n_spin_j, 1)
            # factor (n_det, n_spin_i, n_spin_j)

            # orbital squeeze to remove the atom dimension -1
            return factor * jnp.squeeze(orbital, axis=-1)

    return _compute_orbitals, _sum_orbitals


def anisotropic_orbitals(sigma,
                         orb_vector,
                         d0,
                         activations: Optional[list]=None,
                         basis: Optional[jnp.array]=None,
                         inv_basis: Optional[jnp.array]=None,
                         pbc: bool=False,
                         eps=0.000001,
                         einsum: bool=False):
    # sigma (3, 3 * n_det * n_spin)
    # orb_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det * n_spin_i)
    n_spin = orb_vector.shape[0]
    
    if pbc:
        orb_vector = transform_vector_space(orb_vector, inv_basis)
        orb_vector = jnp.where(orb_vector <= -0.25, -1./(8.*(1. + 2.*(orb_vector + eps))), orb_vector)
        orb_vector = jnp.where(orb_vector >= 0.25, 1./(8.*(1. - 2.*(orb_vector - eps))), orb_vector)
        orb_vector = transform_vector_space(orb_vector, basis)

        # norm = basis.mean()
        # orb_vector = jnp.where(orb_vector <= -0.25 * norm , -1.*norm**2/(8.*(1.*norm + 2.*(orb_vector + eps))), orb_vector)
        # orb_vector = jnp.where(orb_vector >= 0.25 * norm, 1.*norm**2/(8.*(1.*norm - 2.*(orb_vector - eps))), orb_vector)

    if einsum:
        pre_activation = jnp.einsum('jv,vcki->jcki', orb_vector, sigma) + d0
    else:
        pre_activation = jnp.matmul(orb_vector, sigma)  + d0  # n_spin_j, 3 * n_det * n_spin_i
        # the way the activations are unpacked is important, order check in /home/amawi/projects/nn_ansatz/src/scripts/debugging/shapes/sigma_reshape.py
        # surprisingly the alternate way 'works' but there is a difference in performance which is notable at larger system sizes
        pre_activation = pre_activation.reshape(n_spin, 3, -1, order='F')  # order ='F' (n_spin_j, 3, n_spin_i * det), required here because of the 3
    
    exponent = jnp.linalg.norm(pre_activation, axis=1)
    exponential = jnp.exp(-exponent)

    if not activations is None: activations.append(orb_vector)
    return exponential


def isotropic_orbitals(sigma, 
                       orb_vector,                      
                       d0, 
                       activations: Optional[list]=None,
                       orbitals: str = 'isotropic',
                       basis: Optional[jnp.array]=None,
                       inv_basis: Optional[jnp.array]=None,
                       pbc: bool=False,
                       einsum: bool=False,
                       eps=0.000001):
    # sigma (n_det, n_spin_i)
    # ae_vector (n_spin_j, 3)
    # d0 (n_spin_j, n_det, n_spin_i)
    n_spin = orb_vector.shape[0]
    
    if not pbc:
        norm = jnp.linalg.norm(orb_vector, axis=-1, keepdims=True) # (n_spin,)
        exponential = jnp.exp(-norm  * sigma + d0)
    else:
        orb_vector = transform_vector_space(orb_vector, inv_basis)
        # METHOD 1
        if 'sphere' in orbitals:
            norm = jnp.linalg.norm(orb_vector, axis=-1, keepdims=True) # (n_spin,)
            exponential = jnp.exp(-norm * sigma + d0) + jnp.exp(-(1. - norm) * sigma + d0) - 2 * jnp.exp(-sigma * 1. / 2.)
            exponential = jnp.where(norm > 0.5, 0.0, exponential)
        
        # METHOD 2
        if 'spline' in orbitals:
            orb_vector = jnp.where(orb_vector <= -0.25, -1./(8.*(1. + 2.*(orb_vector + eps))), orb_vector)
            orb_vector = jnp.where(orb_vector >= 0.25, 1./(8.*(1. - 2.*(orb_vector - eps))), orb_vector)
            norm = jnp.linalg.norm(orb_vector, axis=-1, keepdims=True) # (n_spin,)
            exponential = jnp.exp(-norm * sigma + d0)

    if not activations is None: activations.append(norm)
    return exponential


def real_plane_wave_orbitals(sigma,
                             orb_vector,
                             d0,
                             activations: Optional[list]=None,
                             kpoints=jnp.array) -> jnp.array:

    # sigma (n_det, n_spin_i, n_atom, 3, 3)
    # orb_vector (n_spin_j, 3)
    # k_points (n_spin_i, 3)
    n_el = orb_vector.shape[0]
    args = orb_vector @ kpoints[:n_el, :].T  # (n_el_j, n_el_i)
    args = split_and_squeeze(args, axis=1)
    pf = [jnp.cos, jnp.sin]
    dets = []
    for i, arg in enumerate(args):
        column = pf[i%2](arg)
        dets.append(column)
    dets = jnp.stack(dets, axis=-1) # (n_el_j, n_el_i)
    return jnp.transpose(dets)[None, ...] # (n_el_i, n_el_j)


def env_pi_i(params: jnp.array,
             factor: jnp.array,
             orbitals: jnp.array,
             activations: list,
             spin: str,
             d0s,
             einsum: bool=False) -> jnp.array:
    
    # params[env_pi...] (m, 1) or (m k i)
    # exponential (j k*i m) or (j k i m)
    # factor (k i j)

    # Einsum sanity check
    n_det, n_spins = factor.shape[:2]

    # n_det * n_spin of (n_spin, n_atom)

    if einsum:
        return factor * (jnp.einsum('jkim,mki->kij', orbitals, params['env_pi_%s' % spin]) + d0s['env_pi_%s' % spin])
    else:
        orbitals = split_and_squeeze(orbitals, axis=1)

        [activations.append(x) for x in orbitals]

        orbitals_sum = []
        for i, e in enumerate(orbitals):
            orbital = (e @ params['env_pi_%s_%i' % (spin, i)]) + d0s['env_pi_%s_%i' % (spin, i)]
            orbitals_sum.append(orbital)
        orbitals_sum = jnp.stack(orbitals_sum, axis=-1) # (j, k*i)

        return factor * jnp.transpose(orbitals_sum.reshape(n_spins, n_det, n_spins), (1, 2, 0))


def logabssumdet_dep(orb_up: jnp.array, orb_down: Optional[jnp.array] = None) -> jnp.array:

    # (k, n_el, n_el)
    if orb_up.shape[-1] == 1:
        s_up, log_up = (jnp.sign(orb_up).squeeze(), jnp.log(jnp.abs(orb_up)).squeeze())
    else: 
        s_up, log_up = jnp.linalg.slogdet(orb_up)
    
    if orb_down is None:
        s_down, log_down = (jnp.ones_like(s_up), jnp.zeros_like(log_up))
    else:
        if orb_down.shape[-1] == 1:
            s_down, log_down = (jnp.sign(orb_down).squeeze(), jnp.log(jnp.abs(orb_down)).squeeze())
        else:
            s_down, log_down = jnp.linalg.slogdet(orb_down)

    logdet_sum = log_up + log_down
    logdet_max = jnp.max(logdet_sum)

    argument = s_up * s_down * jnp.exp(logdet_sum - logdet_max)
    sum_argument = jnp.sum(argument, axis=0)
    sign = jnp.sign(sum_argument)

    log_psi = jnp.log(jnp.abs(sum_argument)) + logdet_max

    return log_psi, sign



def logabssumdet(orb_up: jnp.array, orb_down: Optional[jnp.array] = None) -> jnp.array:
    # Special case if there is only one electron in any channel
    # We can avoid the log(0) issue by not going into the log domain
    
    xs = [orb_up, orb_down] if not orb_down is None else [orb_up]
    dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]
    dets = reduce(
        lambda a, b: a*b, dets
    ) if len(dets) > 0 else 1

    slogdets = [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1]
    maxlogdet = 0
    if len(slogdets) > 0:
        sign_in, logdet = reduce(
            lambda a, b: (a[0]*b[0], a[1]+b[1]), slogdets
        )

        maxlogdet = jnp.max(logdet)
        det = sign_in * dets * jnp.exp(logdet - maxlogdet)
    else:
        det = dets

    result = jnp.sum(det)

    sign_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return log_out, sign_out


def mixer_i(single: jnp.array,
            pairwise: jnp.array,
            n_el: int,
            n_up: int,
            n_down: int,
            pairwise_up_mask,
            pairwise_down_mask):
    # single (n_samples, n_el, n_single_features)
    # pairwise (n_samples, n_pairwise, n_pairwise_features)

    # --- Single summations
    # up
    data_up, data_down = single.split([n_up], axis=0)
    mean_spin_up = jnp.mean(data_up, axis=0, keepdims=True)

    # --- Pairwise summations
    sum_pairwise = jnp.repeat(jnp.expand_dims(pairwise, axis=0), n_el, axis=0)

    # up
    sum_pairwise_up = pairwise_up_mask * sum_pairwise
    sum_pairwise_up = sum_pairwise_up.sum((1, 2)) / float(n_up)

    # down
    if n_down > 0:
        # sum_spin_down = single_down_mask * single
        # sum_spin_down = jnp.sum(sum_spin_down, axis=0, keepdims=True) / float(n_down)
        mean_spin_down = jnp.mean(data_down, axis=0, keepdims=True)

        # down
        sum_pairwise_down = pairwise_down_mask * sum_pairwise
        sum_pairwise_down = sum_pairwise_down.sum((1, 2)) / float(n_down)
    
        single = jnp.concatenate((single, sum_pairwise_up, sum_pairwise_down), axis=1)
        split = jnp.concatenate((mean_spin_up, mean_spin_down), axis=1)
        return single, split
    else:
        single = jnp.concatenate((single, sum_pairwise_up), axis=1)
        return single, mean_spin_up

