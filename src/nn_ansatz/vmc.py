
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad, value_and_grad

import numpy as np
from jax.tree_util import tree_unflatten, tree_flatten

@jit
def sgd(params, grads, lr=1e-4):
    params, tree_map = tree_flatten(params)
    grads, _ = tree_flatten(grads)
    params = [p - lr * g for p, g in zip(params, grads)]
    params = tree_unflatten(tree_map, params)
    return params

@jit
def clip_and_center(e_locs):
    median = jnp.median(e_locs)
    total_var = jnp.mean(jnp.abs(e_locs - median))
    lower, upper = median - 5*total_var, median + 5*total_var
    e_locs = jnp.clip(e_locs, a_min=lower, a_max=upper)
    return e_locs - jnp.mean(e_locs)


def create_grad_function(wf, mol):

    vwf = vmap(wf, in_axes=(None, 0))
    compute_energy = create_energy_function(wf, mol)

    def _grad_function(params, walkers):

        e_locs = lax.stop_gradient(compute_energy(params, walkers))
        e_locs_centered = clip_and_center(e_locs)
        log_psi = vwf(params, walkers)

        return jnp.mean(e_locs_centered * log_psi), e_locs

    return jit(grad(_grad_function, has_aux=True))


def create_atom_batch(r_atoms, n_samples):
    return jnp.repeat(jnp.expand_dims(r_atoms, axis=0), n_samples, axis=0)


def create_energy_function(wf, mol):

    r_atoms, z_atoms = mol.r_atoms, mol.z_atoms

    local_kinetic_energy = jit(vmap(local_kinetic_energy_i(wf), in_axes=(None, 0)))
    compute_potential_energy = jit(vmap(compute_potential_energy_i, in_axes=(0, None, None)))

    def _compute_local_energy(params, walkers):
        potential_energy = compute_potential_energy(walkers, r_atoms, z_atoms)
        kinetic_energy = local_kinetic_energy(params, walkers)
        return potential_energy + kinetic_energy

    return _compute_local_energy


def batched_cdist_l2(x1, x2):

    x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True)
    x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True)
    cdist = jnp.sqrt(jnp.swapaxes(x1_sq, -1, -2) + x2_sq \
                     - jnp.sum(2 * jnp.expand_dims(x1, axis=0) * jnp.expand_dims(x2, axis=1), axis=-1))
    return cdist


def compute_potential_energy_i(walkers, r_atoms, z_atoms):
    n_atom = r_atoms.shape[0]

    e_e_dist = batched_cdist_l2(walkers, walkers)  # electron - electron distances
    potential_energy = jnp.sum(jnp.tril(1. / e_e_dist, k=-1), axis=(-1, -2))

    a_e_dist = batched_cdist_l2(r_atoms, walkers)  # atom - electron distances
    potential_energy -= jnp.sum(z_atoms / a_e_dist)

    if n_atom > 1:  # THIS HAS NOT BEEN VERIFIED
        a_a_dist = batched_cdist_l2(r_atoms, r_atoms)
        weighted_a_a = jnp.einsum('bn,bm,bnm->bnm', z_atoms, z_atoms, 1 / a_a_dist)
        unique_a_a = weighted_a_a[:, jnp.tril(np.ones((n_atom, n_atom), dtype=bool), -1)]  # this will not work
        potential_energy += jnp.sum(unique_a_a, axis=-1)

    return potential_energy


def local_kinetic_energy_i(wf):

    def _lapl_over_f(params, x):
        x = x.reshape(-1)
        n = x.shape[0]
        eye = jnp.eye(n, dtype=x.dtype)
        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[..., i],))
            return val + primal[i]**2 + tangent[i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


compute_potential_energy = jit(vmap(compute_potential_energy_i, in_axes=(0, None, None)))
# local_kinetic_energy = jit(vmap(local_kinetic_energy_i(wf), in_axes=(None, 0)))