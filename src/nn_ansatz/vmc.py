import itertools

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad
from jax.tree_util import tree_unflatten, tree_flatten
from itertools import chain, combinations_with_replacement

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

    vwf = vmap(wf, in_axes=(None, 0, 0))
    compute_energy = create_energy_fn(wf, mol)

    def _grad_function(params, walkers, d0s):

        e_locs = lax.stop_gradient(compute_energy(params, walkers, d0s))
        e_locs_centered = clip_and_center(e_locs)
        log_psi = vwf(params, walkers, d0s)

        return jnp.mean(e_locs_centered * log_psi), e_locs

    return jit(grad(_grad_function, has_aux=True))


def create_atom_batch(r_atoms, n_samples):
    return jnp.repeat(jnp.expand_dims(r_atoms, axis=0), n_samples, axis=0)


def create_energy_fn(wf, mol):

    r_atoms, z_atoms = mol.r_atoms, mol.z_atoms

    local_kinetic_energy = vmap(local_kinetic_energy_i(wf), in_axes=(None, 0, 0))
    compute_potential_energy = create_potential_energy(mol)

    def _compute_local_energy(params, walkers, d0s):
        potential_energy = compute_potential_energy(walkers, r_atoms, z_atoms)
        kinetic_energy = local_kinetic_energy(params, walkers, d0s)
        return potential_energy + kinetic_energy

    return jit(_compute_local_energy)


def all_translations(cv1, cv2, cv3):
    return [cv1, cv1 + cv2, cv1 - cv2, cv1 + cv3, cv1 - cv3,
            cv1 + cv2 + cv3, cv1 + cv2 - cv3, cv1 - cv2 + cv3, cv1 - cv2 - cv3]


def compute_volume(cv1, cv2, cv3):
    cross = jnp.cross(cv2, cv3, axisa=0, axisb=0)
    box = cross @ cv1
    volume = jnp.abs(box.squeeze())
    return volume


def generate_vectors(vector_set, n):
    # lattice vectors
    lattice_vectors = list(
        chain.from_iterable(combinations_with_replacement(vector_set, i) for i in range(1, n + 1)))
    lattice_vectors = [jnp.sum(jnp.concatenate(x, axis=-1), axis=-1) for x in lattice_vectors]
    lattice_vectors = jnp.array([x for x in lattice_vectors if not jnp.sum(jnp.zeros(3, ) == x) == 3])
    return lattice_vectors

def inner(v1, v2):
    return jnp.sum(v1 * v2, axis=-1, keepdims=True)

def create_potential_energy(mol):
    if mol.periodic_boundaries:
        cv1, cv2, cv3 = mol.cell_basis.split(3, axis=1)  # (3, 1) vectors
        # translations = all_translations(cv1, cv2, cv3)
        # translations.extend(all_translations(-cv1, cv2, cv3))
        # translations.extend([cv2, cv2 + cv3, cv2 - cv3, -cv2, -cv2 + cv3, -cv2 - cv3])
        # translations.extend([cv3, -cv3])
        # translation_vectors = jnp.concatenate(translations, axis=-1).transpose()
        # translation_vectors = jnp.expand_dims(translation_vectors, axis=0)
        #
        n = 3
        vector_set = [cv1, -cv1, cv2, -cv2, cv3, -cv3]
        lattice_vectors = generate_vectors(vector_set, n)

        volume = compute_volume(cv1, cv2, cv3)
        kappa = 1.
        constant = jnp.pi / (kappa ** 2 * volume)

        # reciprocal vectors
        rv1 = 2 * jnp.pi * jnp.cross(cv2, cv3, axisa=0, axisb=0) / volume
        rv2 = jnp.pi * jnp.cross(cv3, cv1, axisa=0, axisb=0) / volume
        rv3 = jnp.pi * jnp.cross(cv1, cv2, axisa=0, axisb=0) / volume
        vector_set = [x.transpose() for x in (rv1, -rv1, rv2, -rv2, rv3, -rv3)]
        reciprocal_vectors = generate_vectors(vector_set, n)
        reciprocal_factors = (4*jnp.pi / volume) * \
                             jnp.exp(- inner(reciprocal_vectors, reciprocal_vectors) / (4*kappa**2)) / \
                             inner(reciprocal_vectors, reciprocal_vectors)

        def compute_potential_energy_solid_i(walkers, r_atoms, z_atoms):

            """
            :param walkers (n_el, 3):
            :param r_atoms (n_atoms, 3):
            :param z_atoms (n_atoms, ):

            Pseudocode:
                - compute the potential energy (pe) of the cell
                - compute the pe of the cell electrons with electrons outside
                - compute the pe of the cell electrons with nuclei outside
                - compute the pe of the cell nuclei with nuclei outside
            """

            ex_walkers = (jnp.expand_dims(walkers, axis=1) + lattice_vectors).reshape(-1, 3)  # (n_el * 26, 3)
            ex_r_atoms = (jnp.expand_dims(r_atoms, axis=1) + lattice_vectors).reshape(-1, 3)  # (n_atom * 26, 3)
            ex_z_atoms = jnp.expand_dims(z_atoms, axis=0).repeat(len(lattice_vectors), axis=0)  # (n_atom * 26, 1)

            potential_energy = compute_potential_energy_i(walkers, r_atoms, z_atoms)

            ex_e_e_dist = batched_cdist_l2(walkers, ex_walkers)
            # erfc = lax.erfc(kappa * ex_e_e_dist)
            # potential_energy += 0.5 * jnp.sum(erfc / ex_e_e_dist)
            inner = reciprocal_vectors @
            potential_energy += 0.5 * jnp.sum(1. / ex_e_e_dist)

            ex_a_e_dist = batched_cdist_l2(walkers, ex_r_atoms)
            # erfc = lax.erfc(kappa * ex_a_e_dist)
            # potential_energy -= 0.5 * jnp.sum(erfc * ex_z_atoms / ex_a_e_dist)
            potential_energy -= 0.5 * jnp.sum(ex_z_atoms / ex_a_e_dist)

            ex_a_a_dist = batched_cdist_l2(r_atoms, ex_r_atoms)
            # erfc = lax.erfc(kappa * ex_a_a_dist)
            # potential_energy += 0.5 * jnp.sum(erfc * (z_atoms[None, :] * ex_z_atoms) / ex_a_a_dist)
            potential_energy += 0.5 * jnp.sum((z_atoms[None, :] * ex_z_atoms) / ex_a_a_dist)

            return potential_energy

        # return vmap(compute_potential_energy_i, in_axes=(0, None, None))
        return vmap(compute_potential_energy_solid_i, in_axes=(0, None, None))

    return vmap(compute_potential_energy_i, in_axes=(0, None, None))


def batched_cdist_l2(x1, x2):

    x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True)
    x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True)
    cdist = jnp.sqrt(jnp.swapaxes(x1_sq, -1, -2) + x2_sq \
                     - jnp.sum(2 * jnp.expand_dims(x1, axis=0) * jnp.expand_dims(x2, axis=1), axis=-1))
    return cdist





def compute_potential_energy_i(walkers, r_atoms, z_atoms):
    """

    :param walkers (n_el, 3):
    :param r_atoms (n_atoms, 3):
    :param z_atoms (n_atoms, ):
    :return:

    pseudocode:
        - compute potential energy contributions
            - electron - electron interaction
            - atom - electron interaction
            - atom - atom interation
    """

    n_atom = r_atoms.shape[0]

    e_e_dist = batched_cdist_l2(walkers, walkers)
    potential_energy = jnp.sum(jnp.tril(1. / e_e_dist, k=-1))

    a_e_dist = batched_cdist_l2(r_atoms, walkers)
    potential_energy -= jnp.sum(z_atoms / a_e_dist)

    if n_atom > 1:
        a_a_dist = batched_cdist_l2(r_atoms, r_atoms)
        weighted_a_a = (z_atoms[:, None] * z_atoms[None, :]) / a_a_dist
        unique_a_a = jnp.tril(weighted_a_a, k=-1)
        potential_energy += jnp.sum(unique_a_a)

    return potential_energy


def local_kinetic_energy_i(wf):
    """
    FUNCTION SLIGHTLY ADAPTED FROM DEEPMIND JAX FERMINET IMPLEMTATION
    https://github.com/deepmind/ferminet/tree/jax

    """
    def _lapl_over_f(params, walkers, d0s):
        walkers = walkers.reshape(-1)
        n = walkers.shape[0]
        eye = jnp.eye(n, dtype=walkers.dtype)
        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y, d0s)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (walkers,), (eye[..., i],))
            return val + primal[i]**2 + tangent[i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


compute_potential_energy = jit(vmap(compute_potential_energy_i, in_axes=(0, None, None)))
# local_kinetic_energy = jit(vmap(local_kinetic_energy_i(wf), in_axes=(None, 0)))