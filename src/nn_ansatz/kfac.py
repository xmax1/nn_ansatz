
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad, value_and_grad

import numpy as np
from jax.tree_util import tree_unflatten, tree_flatten

from .vmc import create_grad_function


def kfac(kfac_wf, wf, mol, params, walkers, d0s, lr, damping, norm_constraint):

    kfac_update, substate = create_natural_gradients_fn(kfac_wf, wf, mol, params, walkers, d0s)

    def _get_params(x):
        return x[0]

    def _update(step, grads, state):
        params = _get_params(state)
        params, tree = tree_flatten(params)
        params = [p - g for p, g in zip(params, grads)]
        params = tree_unflatten(tree, params)

        

        return [params, *state[1:]]

    state = [*substate, lr, damping, norm_constraint]

    return _update, _get_params, kfac_update, state


def create_sensitivities_grad_fn(kfac_wf):
    vwf = vmap(kfac_wf, in_axes=(None, 0, 0))

    def _sum_log_psi(params, walkers, d0s):
        log_psi, activations = vwf(params, walkers, d0s)
        return log_psi.mean()

    grad_fn = jit(grad(_sum_log_psi, argnums=2))

    return grad_fn


def create_natural_gradients_fn(kfac_wf, wf, mol, params, walkers, d0s):

    sensitivities_fn = create_sensitivities_grad_fn(kfac_wf)
    # grad_fn = create_grad_function(wf, mol)
    vwf = jit(vmap(kfac_wf, in_axes=(None, 0, 0)))

    @jit
    def _kfac_step(step, gradients, activations, sensitivities, state):

        params, maas, msss, lr, damping, norm_constraint = state

        gradients, gradients_tree_map = tree_flatten(gradients)
        activations, activations_tree_map = tree_flatten(activations)
        sensitivities, sensitivities_tree_map = tree_flatten(sensitivities)

        ngs = []
        new_maas = []
        new_msss = []
        for g, a, s, maa, mss in zip(gradients, activations, sensitivities, maas, msss):

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

            maa, mss = update_maa_and_mss(step, maa, aa, mss, ss)

            dmaa, dmss = damp(maa, mss, sl_factor, damping)

            # chol_dmaa = jnp.linalg.cholesky(dmaa)
            # chol_dmss = jnp.linalg.cholesky(dmss)

            # chol_dmaa = jax.scipy.linalg.cho_factor(dmaa)
            # chol_dmss = jax.scipy.linalg.cho_factor(dmss)

            # inv_dmaa = jax.scipy.linalg.cho_solve(chol_dmaa, jnp.ones((maa.shape[0], maa.shape[0])))  # , check_finite=False for performance
            # inv_dmss = jax.scipy.linalg.cho_solve(chol_dmss, jnp.ones((mss.shape[0], mss.shape[0])))

            # ng = inv_dmaa @ g @ inv_dmss / sl_factor

            vals_dmaa, vecs_dmaa = jnp.linalg.eigh(dmaa)
            vals_dmss, vecs_dmss = jnp.linalg.eigh(dmss)

            tmp = (jnp.transpose(vecs_dmaa) @ g @ vecs_dmss) / (vals_dmaa[:, None] * vals_dmss[None, :])
            ng = vecs_dmaa @ tmp @ jnp.transpose(vecs_dmss)

            ngs.append(ng)
            new_maas.append(maa)
            new_msss.append(mss)

        eta = compute_norm_constraint(ngs, gradients, lr, norm_constraint)

        return [lr * eta * ng / sl_factor for ng in ngs], (new_maas, new_msss, lr, damping, norm_constraint)

    def _compute_natural_gradients(step, grads, state, walkers, d0s):

        params = state[0]
        # yes there is a more efficient way of doing this.
        # It can be reduced by 1 forward passes and 1 backward pass
        _, activations = vwf(params, walkers, d0s)
        sensitivities = sensitivities_fn(params, walkers, d0s)

        ngs, state = _kfac_step(step, grads, activations, sensitivities, state)

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


def update_maa_and_mss(step, maa, aa, mss, ss):
    cov_moving_weight = jnp.min(jnp.array([step, 0.95]))
    cov_instantaneous_weight = 1. - cov_moving_weight
    total = cov_moving_weight + cov_instantaneous_weight

    maa = (cov_moving_weight * maa + cov_instantaneous_weight * aa) / total
    mss = (cov_moving_weight * mss + cov_instantaneous_weight * ss) / total

    return maa, mss
