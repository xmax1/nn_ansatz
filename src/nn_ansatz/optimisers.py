
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad, value_and_grad, pmap

import numpy as np
import os
from jax.tree_util import tree_unflatten, tree_flatten

from .vmc import create_grad_function


def update_maa_and_mss(step, maa, aa, mss, ss):
    cov_moving_weight = jnp.min(jnp.array([step, 0.95]))
    cov_instantaneous_weight = 1. - cov_moving_weight
    total = cov_moving_weight + cov_instantaneous_weight

    maa = (cov_moving_weight * maa + cov_instantaneous_weight * aa) / total
    mss = (cov_moving_weight * mss + cov_instantaneous_weight * ss) / total

    return maa, mss


def kfac(kfac_wf, wf, mol, params, walkers, d0s, lr, damping, norm_constraint, **kwargs):
    kfac_update, substate = create_natural_gradients_fn(kfac_wf, wf, mol, params, walkers, d0s)

    def _get_params(state):
        return state[0]

    def _update(step, grads, state):
        params = _get_params(state)
        params, tree = tree_flatten(params)
        params = [p - g for p, g in zip(params, grads)]
        params = tree_unflatten(tree, params)

        return [params, *state[1:]]

    state = [*substate, lr, damping, norm_constraint]

    if os.environ.get('JIT') == 'True':
        return jit(_update), jit(_get_params), kfac_update, state
    return _update, _get_params, kfac_update, state


def create_sensitivities_grad_fn(kfac_wf):
    vwf = vmap(kfac_wf, in_axes=(None, 0, 0))

    def _sum_log_psi(params, walkers, d0s):
        log_psi, activations = vwf(params, walkers, d0s)
        return log_psi.mean()

    grad_fn = grad(_sum_log_psi, argnums=2)

    return grad_fn


def create_natural_gradients_fn(kfac_wf, wf, mol, params, walkers, d0s):
    sensitivities_fn = create_sensitivities_grad_fn(kfac_wf)
    vwf = vmap(kfac_wf, in_axes=(None, 0, 0))

    def _kfac_step(step, gradients, aas, sss, maas, msss, sl_factors, lr, damping, norm_constraint):

        gradients, gradients_tree_map = tree_flatten(gradients)
        ngs = []
        new_maas = []
        new_msss = []
        for g, aa, ss, maa, mss, sl_factor in zip(gradients, aas, sss, maas, msss, sl_factors):

            maa, mss = update_maa_and_mss(step, maa, aa, mss, ss)

            dmaa, dmss = damp(maa, mss, sl_factor, damping)

            # chol_dmaa = jnp.linalg.cholesky(dmaa)
            # chol_dmss = jnp.linalg.cholesky(dmss)

            dmaa = (dmaa + jnp.transpose(dmaa)) / 2.
            dmss = (dmss + jnp.transpose(dmss)) / 2.

            chol_dmaa = jax.scipy.linalg.cho_factor(dmaa)
            chol_dmss = jax.scipy.linalg.cho_factor(dmss)

            inv_dmaa = jax.scipy.linalg.cho_solve(chol_dmaa, jnp.eye(maa.shape[0]))  # , check_finite=False for performance
            inv_dmss = jax.scipy.linalg.cho_solve(chol_dmss, jnp.eye(mss.shape[0]))

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

    def _compute_covariances(activations, sensitivities):

        activations, activations_tree_map = tree_flatten(activations)
        sensitivities, sensitivities_tree_map = tree_flatten(sensitivities)

        aas = []
        sss = []
        sl_factors = []
        for a, s in zip(activations, sensitivities):
            n = a.shape[0]
            sl_factor = 1.

            if len(a.shape) == 3:
                sl_factor = float(a.shape[1] ** 2)
                a = a.mean(1)
            if len(s.shape) == 3:
                sl_factor = float(s.shape[1] ** 2)
                s = s.mean(1)

            aa = jnp.transpose(a) @ a / float(n)
            ss = jnp.transpose(s) @ s / float(n)

            aas.append(aa)
            sss.append(ss)
            sl_factors.append(sl_factor)

        return aas, sss, sl_factors
    
    if not os.environ.get('no_JIT') == 'True':
        _kfac_step = jit(_kfac_step)
        vwf = jit(vwf)
        sensitivities_fn = jit(sensitivities_fn)
        _compute_covariances = jit(_compute_covariances)
    
    vwf = pmap(vwf, in_axes=(None, 0, 0))
    sensitivities_fn = pmap(sensitivities_fn, in_axes=(None, 0, 0))
    _compute_covariances = pmap(_compute_covariances, in_axes=(0, 0))

    def _compute_natural_gradients(step, grads, state, walkers, d0s):

        params, maas, msss, lr, damping, norm_constraint = state

        _, activations = vwf(params, walkers, d0s)
        sensitivities = sensitivities_fn(params, walkers, d0s)
        aas, sss, sl_factors = _compute_covariances(activations, sensitivities)
        aas = [jax.device_put(aa, jax.devices()[0]).mean(0) for aa in aas]
        sss = [jax.device_put(ss, jax.devices()[0]).mean(0) for ss in sss]
        sl_factors = [s[0] for s in sl_factors]

        ngs, state = _kfac_step(step, grads, aas, sss, maas, msss, sl_factors, lr, damping, norm_constraint)

        return ngs, (params, *state)

    _, activations = vwf(params, walkers, d0s)
    sensitivities = sensitivities_fn(params, walkers, d0s)

    activations, activations_tree_map = tree_flatten(activations)
    sensitivities, sensitivities_tree_map = tree_flatten(sensitivities)
    maas = [jnp.zeros((a.shape[-1], a.shape[-1])) for a in activations]
    msss = [jnp.zeros((s.shape[-1], s.shape[-1])) for s in sensitivities]
    substate = (params, maas, msss)

    return _compute_natural_gradients, substate


def check_symmetric(x):
    x = x - x.transpose(-1, -2)
    print(x.mean())


def compute_norm_constraint(nat_grads, grads, lr, norm_constraint):
    sq_fisher_norm = 0.
    for ng, g in zip(nat_grads, grads):
        sq_fisher_norm += (ng * g).sum()
    eta = jnp.min(jnp.array([1., jnp.sqrt(norm_constraint / (lr**2 * sq_fisher_norm))]))
    return eta


def decay_variable(self, variable, iteration):
    return variable / (1. + self.decay * iteration)


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

    maa += eye_a * m_aa_damping
    mss += eye_s * m_ss_damping
    return maa, mss


def get_tr_norm(x):
    trace = jnp.diagonal(x).sum(-1)
    return jnp.max(jnp.array([1e-5, trace]))



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
