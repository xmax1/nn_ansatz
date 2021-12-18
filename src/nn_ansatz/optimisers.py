from .parameters import initialise_d0s
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad, value_and_grad, pmap
from jax.tree_util import tree_unflatten, tree_flatten

import numpy as np
import os
from functools import partial

from .ansatz import create_wf


def kfac(mol, params, walkers, lr, damping, norm_constraint):

    kfac_update, substate = create_natural_gradients_fn(mol, params, walkers)

    def _get_params(state):
        return state[0]

    @jit
    def _update(step, grads, state):
        params, _, _, lr, damping, norm_constraint = state
        params, tree = tree_flatten(params)
        params = [p - g for p, g in zip(params, grads)]
        params = tree_unflatten(tree, params)

        return [params, *state[1:3], lr, damping, norm_constraint]

    state = [*substate, lr, damping, norm_constraint]

    return _update, _get_params, kfac_update, state


def create_natural_gradients_fn(mol, params, walkers):

    kfac_wf = create_wf(mol, kfac=True)
    
    d0s = initialise_d0s(mol, expand=True)  # expand adds leading dimensions for multiple devices

    def sum_log_psi(params, walkers, d0s):
        log_psi, activations = kfac_wf(params, walkers, d0s)
        return log_psi.sum(), activations

    def compute_covariances(activations, sensitivities):  # (m, sl, nf)
        activations, _ = tree_flatten(activations)
        sensitivities, _ = tree_flatten(sensitivities)

        aas, sss = [], []
        for a, s in zip(activations, sensitivities):
            # a -= a.mean(1, keepdims=True)
            # s -= s.mean(1, keepdims=True)

            a = a.reshape(-1, a.shape[-1])
            s = s.reshape(-1, s.shape[-1])

            # a -= a.mean(0, keepdims=True)
            # s -= s.mean(0, keepdims=True)

            aa = (jnp.transpose(a) @ a) / float(a.shape[0])
            ss = (jnp.transpose(s) @ s) / float(s.shape[0])

            aas.append(aa)
            sss.append(ss)

        return aas, sss

    compute_sensitivities = grad(sum_log_psi, argnums=2, has_aux=True)

    sl_factor_idx = 1
    if bool(os.environ.get('DISTRIBUTE')) is True:
        compute_sensitivities = pmap(compute_sensitivities, in_axes=(None, 0, 0))
        compute_covariances = pmap(compute_covariances, in_axes=(0, 0))
        sl_factor_idx += 1
    
    sensitivities, activations = compute_sensitivities(params, walkers, d0s)
    activations, _ = tree_flatten(activations)
    sensitivities, _ = tree_flatten(sensitivities)
    maas = [jnp.zeros((a.shape[-1], a.shape[-1])) for a in activations]
    msss = [jnp.zeros((s.shape[-1], s.shape[-1])) for s in sensitivities]
    sl_factors = [a.shape[sl_factor_idx] if len(a.shape) > (sl_factor_idx + 1) else 1. for a in activations]
    print('sl_factors: \n', sl_factors)
    substate = (params, maas, msss)

    @jit
    def compute_natural_gradients_local(step, gradients, aas, sss, maas, msss, lr, damping, norm_constraint):

        if bool(os.environ.get('DISTRIBUTE')) is True:
            aas = [aa.mean(0) for aa in aas]
            sss = [ss.mean(0) for ss in sss]

            # lr = decay_variable(lr, step)
        damping_current = decay_variable(damping, step, decay=1e-2, floor=1e-6)
        norm_constraint_current = decay_variable(norm_constraint, step, decay=1e-4, floor=1e-6)

        gradients, _ = tree_flatten(gradients)
        
        ngs, new_maas, new_msss = [], [], []
        for g, aa, ss, maa, mss, sl_factor in zip(gradients, aas, sss, maas, msss, sl_factors):
            sl_factor = 1.

            maa, mss = update_maa_and_mss(step, maa, aa, mss, ss, sl_factor)

            dmaa, dmss = damp(maa, mss, sl_factor, damping_current)

            # dmaa = maa + (jnp.eye(maa.shape[0]) * damping_current)
            # dmss = mss + (jnp.eye(mss.shape[0]) * damping_current)

            dmaa = (dmaa + jnp.transpose(dmaa)) / 2.
            dmss = (dmss + jnp.transpose(dmss)) / 2.

            chol_dmaa = jax.scipy.linalg.cho_factor(dmaa)
            chol_dmss = jax.scipy.linalg.cho_factor(dmss)

            inv_dmaa = jax.scipy.linalg.cho_solve(chol_dmaa, jnp.eye(maa.shape[0]), check_finite=False)  # , check_finite=False for performance
            inv_dmss = jax.scipy.linalg.cho_solve(chol_dmss, jnp.eye(mss.shape[0]), check_finite=False)

            ng = inv_dmaa @ g @ inv_dmss 

            ngs.append(ng)
            new_maas.append(maa)
            new_msss.append(mss)

        eta = compute_norm_constraint(ngs, gradients, lr, norm_constraint_current)

        return [lr * eta * ng for ng in ngs], (new_maas, new_msss, lr, damping, norm_constraint)


    def compute_natural_gradients(step, grads, state, walkers):

        params, maas, msss, lr, damping, norm_constraint = state

        sensitivities, activations = compute_sensitivities(params, walkers, d0s)
        aas, sss = compute_covariances(activations, sensitivities)

        grads, state = compute_natural_gradients_local(step, grads, aas, sss, maas, msss, lr, damping, norm_constraint)

        return grads, (params, *state)
    
    return compute_natural_gradients, substate


def create_natural_gradients_fn2(mol, params, walkers):
    
    kfac_wf = create_wf(mol, kfac=True)
    
    d0s = initialise_d0s(mol, expand=True)  # expand adds leading dimensions for multiple devices

    _sensitivities_fn = create_sensitivities_grad_fn(kfac_wf)

    _compute_covariances = compute_covariances  # weird workaround, something to do with the namespaces idm
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        kfac_wf = pmap(kfac_wf, in_axes=(None, 0, 0))
        _sensitivities_fn = pmap(_sensitivities_fn, in_axes=(None, 0, 0))
        _compute_covariances = pmap(_compute_covariances, in_axes=(0, 0))

    # generate the initial moving averages
    # can be done with shapes but requires thinking and we don't do thinking
    _, activations = kfac_wf(params, walkers, d0s)
    sensitivities = _sensitivities_fn(params, walkers, d0s)
    activations, activations_tree_map = tree_flatten(activations)
    sensitivities, sensitivities_tree_map = tree_flatten(sensitivities)
    maas = [jnp.zeros((a.shape[-1], a.shape[-1])) for a in activations]
    msss = [jnp.zeros((s.shape[-1], s.shape[-1])) for s in sensitivities]
    substate = (params, maas, msss)
    sl_factors = get_sl_factors(activations)

    _compute_natural_gradients = partial(compute_natural_gradients, 
                                         sl_factors=sl_factors,
                                         d0s=d0s,
                                         kfac_wf=kfac_wf, 
                                         _compute_covariances=_compute_covariances,
                                         _sensitivities_fn=_sensitivities_fn)

    

    return jit(_compute_natural_gradients), substate


def get_sl_factors(activations):
    ''' function to extract the spatial location factors (n_uses of weights) from activations '''
    correction = 0
    if bool(os.environ.get('DISTRIBUTE')) is True:
        correction = -1
    sl_factors = []
    for a in activations:
        lena = len(a.shape) + correction
        sl_factor = 1.
        if lena == 3:
            sl_factor = a.shape[-2]
        sl_factors.append(sl_factor**2)
        # print('sl_factors: ', a.shape, sl_factor)
    return sl_factors


def update_maa_and_mss(step, maa, aa, mss, ss, sl_factor, cov_moving_weight=0.8):

    cov_moving_weight = jnp.min(jnp.array([step, cov_moving_weight]))

    cov_instantaneous_weight = 1. - cov_moving_weight
    total = cov_moving_weight + cov_instantaneous_weight

    maa = (cov_moving_weight * maa + cov_instantaneous_weight * aa) / total

    ss /= sl_factor
    mss = (cov_moving_weight * mss + cov_instantaneous_weight * ss) / total

    return maa, mss


def create_sensitivities_grad_fn(kfac_wf):

    def _mean_log_psi(params, walkers, d0s):
        log_psi, _ = kfac_wf(params, walkers, d0s)
        return log_psi.sum()  # is this sum? making it the mean improves the optimisation a lot (puzzled face)

    grad_fn = grad(_mean_log_psi, argnums=2)
    return grad_fn


def compute_covariances(activations, sensitivities):

    activations, activations_tree_map = tree_flatten(activations)
    sensitivities, sensitivities_tree_map = tree_flatten(sensitivities)

    aas = []
    sss = []
    for a, s in zip(activations, sensitivities):
        # shapes (m, sl_factor, n_f)
        print(a.shape, s.shape)
        n = a.shape[0]
        if len(a.shape) == 3:
            a = a.mean(1)
        if len(s.shape) == 3:
            s = s.mean(1)

        aa = (jnp.transpose(a) @ a) / float(n)
        ss = (jnp.transpose(s) @ s) / float(n)

        aas.append(aa)
        sss.append(ss)

    return aas, sss


# walkers (n_gpus, n_walkers, n_el, 3)

def compute_natural_gradients(step, grads, state, walkers, d0s, sl_factors, kfac_wf, _compute_covariances, _sensitivities_fn):

    params, maas, msss, lr, damping, norm_constraint = state

    _, activations = kfac_wf(params, walkers, d0s)
    sensitivities = _sensitivities_fn(params, walkers, d0s)
    aas, sss = _compute_covariances(activations, sensitivities)

    if bool(os.environ.get('DISTRIBUTE')) is True:
        [print(aa.shape) for aa in aas]
        [print(ss.shape) for ss in sss]
        aas = [aa.mean(0) for aa in aas]
        sss = [ss.mean(0) for ss in sss]

    ngs, state = kfac_step(step, grads, aas, sss, maas, msss, sl_factors, lr, damping, norm_constraint)

    return ngs, (params, *state)

def kfac_step(step, gradients, aas, sss, maas, msss, sl_factors, lr, damping, norm_constraint):

    # lr = decay_variable(lr, step)
    damping = decay_variable(damping, step, decay=1e-2, floor=1e-6)
    norm_constraint = decay_variable(norm_constraint, step, decay=1e-4, floor=1e-6)

    gradients, _ = tree_flatten(gradients)
    ngs = []
    new_maas = []
    new_msss = []

    for g, aa, ss, maa, mss, sl_factor in zip(gradients, aas, sss, maas, msss, sl_factors):
        # print(g.shape, aa.shape, ss.shape, sl_factor)

        # print(g.shape, aa.shape, ss.shape, maa.shape, mss.shape)

        maa, mss = update_maa_and_mss(step, maa, aa, mss, ss)

        dmaa, dmss = damp(maa, mss, sl_factor, damping)

        # chol_dmaa = jnp.linalg.cholesky(dmaa)
        # chol_dmss = jnp.linalg.cholesky(dmss)

        dmaa = (dmaa + jnp.transpose(dmaa)) / 2.
        dmss = (dmss + jnp.transpose(dmss)) / 2.

        chol_dmaa = jax.scipy.linalg.cho_factor(dmaa)
        chol_dmss = jax.scipy.linalg.cho_factor(dmss)

        inv_dmaa = jax.scipy.linalg.cho_solve(chol_dmaa, jnp.eye(maa.shape[0]), check_finite=False)  # , check_finite=False for performance
        inv_dmss = jax.scipy.linalg.cho_solve(chol_dmss, jnp.eye(mss.shape[0]), check_finite=False)

        # the zero index takes the values on device 0
        ng = inv_dmaa @ g @ inv_dmss / sl_factor

        # vals_dmaa, vecs_dmaa = jnp.linalg.eigh(dmaa)
        # vals_dmss, vecs_dmss = jnp.linalg.eigh(dmss)
        #
        # tmp = (jnp.transpose(vecs_dmaa) @ g @ vecs_dmss) / (vals_dmaa[:, None] * vals_dmss[None, :])
        # ng = vecs_dmaa @ tmp @ jnp.transpose(vecs_dmss)

        ngs.append(ng)
        new_maas.append(maa)
        new_msss.append(mss)

    eta = compute_norm_constraint(ngs, gradients, lr, norm_constraint)

    return [lr * eta * ng for ng in ngs], (new_maas, new_msss, lr, damping, norm_constraint)


def check_symmetric(x):
    x = x - x.transpose(-1, -2)


def compute_norm_constraint(nat_grads, grads, lr, norm_constraint):
    sq_fisher_norm = 0.
    for ng, g in zip(nat_grads, grads):
        sq_fisher_norm += (ng * g).sum()
    eta = jnp.min(jnp.array([1., jnp.sqrt(norm_constraint / (lr**2 * sq_fisher_norm))]))
    return eta


def decay_variable(variable, iteration, decay=1e-2, floor=1e-6):
    return jnp.clip(variable / (1. + decay * iteration), a_min=floor)


def damp(maa, mss, sl_factor, damping):

    dim_a = maa.shape[-1]
    dim_s = mss.shape[-1]

    tr_a = get_tr_norm(maa)
    tr_s = get_tr_norm(mss)

    pi = ((tr_a * dim_s) / (tr_s * dim_a))

    eye_a = jnp.eye(dim_a, dtype=maa.dtype)
    eye_s = jnp.eye(dim_s, dtype=maa.dtype)

    m_aa_damping = jnp.sqrt((pi * damping / sl_factor))
    m_ss_damping = jnp.sqrt((damping / (pi * sl_factor)))

    maa += eye_a * jnp.clip(m_aa_damping, a_min=1e-6, a_max=1e-1)
    mss += eye_s * jnp.clip(m_ss_damping, a_min=1e-6, a_max=1e-1)
    return maa, mss


def get_tr_norm(x):
    trace = jnp.diagonal(x).sum(-1)
    return trace



def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0

  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params
