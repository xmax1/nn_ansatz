import itertools
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad, pmap
from jax.tree_util import tree_unflatten, tree_flatten
from itertools import chain, combinations, combinations_with_replacement, product

import math
from jax.scipy.special import erfc


def create_grad_function(mol, vwf):
    
    compute_energy = create_energy_fn(mol, vwf)

    def _forward_pass(params, walkers):
        e_locs = lax.stop_gradient(compute_energy(params, walkers))

        e_locs_centered = clip_and_center(e_locs) # takes the mean of the data on each device and does not distribute
        log_psi = vwf(params, walkers)

        return jnp.mean(e_locs_centered * log_psi), e_locs

    _param_grad_fn = grad(_forward_pass, has_aux=True)  # has_aux indicates the number of outputs is greater than 1
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        _param_grad_fn = pmap(_param_grad_fn, in_axes=(None, 0))

    '''nb: it is not possible to undevice variables within a pmap'''

    def _grad_fn(params, walkers):
        grads, e_locs = _param_grad_fn(params, walkers)
        grads = jax.device_put(grads, jax.devices()[0])
        grads, tree = tree_flatten(grads)
        grads = [g.mean(0) for g in grads]
        grads = tree_unflatten(tree, grads)
        return grads, jax.device_put(e_locs, jax.devices()[0]).reshape(-1)

    return jit(_grad_fn)


def create_atom_batch(r_atoms, n_samples):
    return jnp.repeat(jnp.expand_dims(r_atoms, axis=0), n_samples, axis=0)


def create_energy_fn(mol, vwf, separate=False):

    local_kinetic_energy = create_local_kinetic_energy(vwf)
    compute_potential_energy = create_potential_energy(mol)

    def _compute_local_energy(params, walkers):
        potential_energy = compute_potential_energy(walkers, mol.r_atoms, mol.z_atoms)
        kinetic_energy = local_kinetic_energy(params, walkers)
        return potential_energy + kinetic_energy if not mol.periodic_boundaries else (potential_energy + kinetic_energy) / mol.n_atoms

    def _compute_local_energy_separate(params, walkers):
        potential_energy = compute_potential_energy(walkers, mol.r_atoms, mol.z_atoms)
        kinetic_energy = local_kinetic_energy(params, walkers)
        return potential_energy, kinetic_energy

    if separate:
        return jit(_compute_local_energy_separate)
    return jit(_compute_local_energy)


def create_local_kinetic_energy(vwf):
    ''' kinetic energy function which works on a vmapped wave function '''

    def _lapl_over_f(params, walkers):
        
        n_walkers = walkers.shape[0]
        
        walkers = walkers.reshape(n_walkers, -1)
        
        n = walkers.shape[-1]
        
        eye = jnp.eye(n, dtype=walkers.dtype)[None, ...].repeat(n_walkers, axis=0)
        
        wf_new = lambda walkers: vwf(params, walkers).sum()
        grad_f = jax.grad(wf_new)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f, (walkers,), (eye[..., i],))  # primal / tangent first / second order derivatives
            return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()

        # from lower to upper (lower, upper, func(int, a) -> a, init_val)
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, jnp.zeros(walkers.shape[0]))

    return _lapl_over_f


def create_potential_energy(mol):
    """

    Notes:
        - May need to shift the origin to the center to enforce the spherical sum condition
        - I am now returning to length of unit cell units which is different to the unit cell length I was using before. How does this affect the computation?
        - Is the reciprocal height computed in the correct way?
    """

    if mol.periodic_boundaries:

        real_basis = mol.real_basis
        reciprocal_basis = mol.reciprocal_basis
        kappa = mol.kappa
        volume = mol.volume

        real_lattice = generate_lattice(real_basis, mol.real_cut)  # (n_lattice, 3)
        reciprocal_lattice = generate_lattice(reciprocal_basis, mol.reciprocal_cut)
        rl_inner_product = inner(reciprocal_lattice, reciprocal_lattice)
        rl_factor = (4.*jnp.pi / volume) * jnp.exp(-rl_inner_product / (4.*kappa**2)) / rl_inner_product  

        e_charges = jnp.array([-1. for i in range(mol.n_el)])
        charges = jnp.concatenate([mol.z_atoms, e_charges], axis=0)  # (n_particle, )
        q_q = charges[None, :] * charges[:, None]  # q_i * q_j  (n_particle, n_particle)

        _compute_potential_energy_solid_i = partial(compute_potential_energy_solid_i, 
                                                    kappa=kappa, real_lattice=real_lattice, reciprocal_lattice=reciprocal_lattice,
                                                    q_q=q_q, charges=charges, volume=volume, rl_factor=rl_factor)

        return vmap(_compute_potential_energy_solid_i, in_axes=(0, None, None))

    return vmap(compute_potential_energy_i, in_axes=(0, None, None))


def compute_potential_energy_solid_i(walkers, 
                                     r_atoms, 
                                     z_atoms, 
                                     kappa, 
                                     real_lattice, 
                                     reciprocal_lattice, 
                                     q_q, 
                                     charges, 
                                     volume, 
                                     rl_factor, 
                                     decompose=False):

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

    # put the walkers and r_atoms together
    walkers = jnp.concatenate([r_atoms, walkers], axis=0)  # (n_particle, 3)

    # compute the Rs0 term
    p_p_vectors = vector_sub(walkers, walkers) # (n_particle, n_particle, 3)
    p_p_distances = compute_distances(walkers, walkers) # (n_particle, n_particle)
    # p_p_distances[p_p_distances < 1e-16] = 1e200  # doesn't matter, diagonal dropped via tril, this is just here to suppress the error
    Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # (n_particle, n_particle) everything above and including the diagonal is zero

    # compute the Rs > 0 term
    ex_walkers = vector_add(walkers, real_lattice)  # (n_particle, n_lattice, 3)
    tmp = walkers[:, None, None, :] - ex_walkers[None, ...]  # (n_particle, n_particle, n_lattice, 3)
    ex_distances = jnp.linalg.norm(tmp, axis=-1)
    # ex_distances = jnp.sqrt(jnp.sum(tmp**2, axis=-1))  
    Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
    real_sum = (q_q * (Rs0 + 0.5 * Rs1)).sum((-1, -2))  # Rs0 no half because of previous tril
    
    # compute the constant factor
    self_interaction = - 0.5 * jnp.diag(q_q * 2 * kappa / jnp.sqrt(jnp.pi)).sum()
    constant = - 0.5 * charges.sum()**2 * jnp.pi / (kappa**2 * volume)  # is zero in neutral case

    # compute the reciprocal term reuse the ee vectors
    exp = jnp.real(jnp.sum(rl_factor[None, None, :] * jnp.exp(1j * p_p_vectors @ jnp.transpose(reciprocal_lattice)), axis=-1))
    reciprocal_sum = 0.5 * (q_q * exp).sum((-1,-2))
    
    potential = real_sum + reciprocal_sum + constant + self_interaction
    if decompose:
        return potential, real_sum, reciprocal_sum, constant, self_interaction
    return potential



def compute_potential_energy_solid_i_v2(walkers, 
                                     r_atoms, 
                                     z_atoms, 
                                     kappa, 
                                     real_lattice, 
                                     reciprocal_lattice, 
                                     q_q, 
                                     charges, 
                                     volume, 
                                     rl_factor, 
                                     decompose=False):

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

    # put the walkers and r_atoms together
    walkers = jnp.concatenate([r_atoms, walkers], axis=0)  # (n_particle, 3)

    # compute the Rs0 term
    p_p_vectors = walkers[None, ...] - walkers[:, None, :]
    p_p_distances = jnp.linalg.norm(p_p_vectors, axis=-1) # (n_particle, n_particle)
    Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # (n_particle, n_particle) everything above and including the diagonal is zero

    # compute the Rs > 0 term
    ex_walkers = walkers[:, None, :] + real_lattice[None, ...] # (n_particle, n_lattice, 3)
    ex_distances = jnp.linalg.norm(walkers[:, None, None, :] - ex_walkers[None, ...], axis=-1)  # (n_particle, n_particle, n_lattice)
    Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
    
    real_sum = (q_q * (Rs0 + 0.5 * Rs1)).sum((-1, -2))  # Rs0 no half because of previous tril

    # compute the reciprocal term reuse the ee vectors
    exp = jnp.real(jnp.sum(rl_factor[None, None, :] * jnp.exp(1j * p_p_vectors @ jnp.transpose(reciprocal_lattice)), axis=-1))
    reciprocal_sum = 0.5 * (q_q * exp).sum((-1,-2))

    # compute the constant factor
    self_interaction = - 0.5 * jnp.diag(q_q * 2 * kappa / jnp.sqrt(jnp.pi)).sum()
    constant = - 0.5 * charges.sum()**2 * jnp.pi / (kappa**2 * volume)  # is zero in neutral case
    
    potential = real_sum + reciprocal_sum + constant + self_interaction
    if decompose:
        return potential, real_sum, reciprocal_sum, constant, self_interaction
    return potential


# OPERATIONAL FUNCTIONS

def vector_sub(v1, v2, axis=-2):
    return jnp.expand_dims(v1, axis=axis) - jnp.expand_dims(v2, axis=axis-1)


def vector_add(v1, v2, axis=-2):
    return jnp.expand_dims(v1, axis=axis) + jnp.expand_dims(v2, axis=axis-1)


def compute_distances(v1, v2):
    inter_vector = vector_sub(v1, v2)
    return jnp.sqrt(jnp.sum(inter_vector**2, axis=-1))


def inner(v1, v2):
    return jnp.sum(v1 * v2, axis=-1)


@jit
def sgd(params, grads, lr=1e-4):
    params, tree_map = tree_flatten(params)
    grads, _ = tree_flatten(grads)
    params = [p - lr * g for p, g in zip(params, grads)]
    params = tree_unflatten(tree_map, params)
    return params


def clip_and_center(e_locs):
    median = jnp.median(e_locs)
    total_var = jnp.mean(jnp.abs(e_locs - median))
    lower, upper = median - 5 * total_var, median + 5 * total_var
    e_locs = jnp.clip(e_locs, a_min=lower, a_max=upper)
    return e_locs - jnp.mean(e_locs)


def fast_generate_lattice(basis, cut):
    
    img_range = jnp.arange(-cut, cut+1)  # x2 to create sphere
    img_sets = list(product(*[img_range, img_range, img_range]))
    # first axis is the number of lattice vectors, second is the integers to scale the primitive vectors, third is the resulting set of vectors
    # then sum over those
    # print(len(img_sets))
    img_sets = jnp.concatenate([jnp.array(x)[None, :, None] for x in img_sets if not jnp.sum(jnp.array(x) == 0) == 3], axis=0)
    # print(img_sets.shape)
    imgs = jnp.sum(img_sets * basis, axis=1)

    # if a sphere around the image is within rcut then keep it
    # lengths = jnp.linalg.norm(imgs, axis=-1)
    # mask = lengths < (cut * len0)
    # img = imgs[mask]
    return imgs


def generate_lattice(basis, cut):
    len0 = jnp.linalg.norm(basis, axis=-1).mean()  # get the mean length of the basis vectors
    
    img_range = jnp.arange(-cut, cut+1)  # x2 to create sphere
    img_sets = list(product(*[img_range, img_range, img_range]))
    # first axis is the number of lattice vectors, second is the integers to scale the primitive vectors, third is the resulting set of vectors
    # then sum over those
    # print(len(img_sets))
    img_sets = jnp.concatenate([jnp.array(x)[None, :, None] for x in img_sets if not jnp.sum(jnp.array(x) == 0) == 3], axis=0)
    # print(img_sets.shape)
    imgs = jnp.sum(img_sets * basis, axis=1)

    # if a sphere around the image is within rcut then keep it
    # lengths = jnp.linalg.norm(imgs, axis=-1)
    # mask = lengths < (cut * len0)
    # img = imgs[mask]
    return imgs






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


# DEPRECIATED FUNCTIONS


def local_kinetic_energy_i(wf):
    """
    FUNCTION SLIGHTLY ADAPTED FROM DEEPMIND JAX FERMINET IMPLEMTATION
    https://github.com/deepmind/ferminet/tree/jax

    kinetic energy function which works when vmapped and the input is an unvmapped wf

    """
    def _lapl_over_f(params, walkers, d0s):

        walkers = walkers.reshape(-1)
        n = walkers.shape[-1]
        eye = jnp.eye(n, dtype=walkers.dtype)

        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y, d0s)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (walkers,), (eye[..., i],))
            return val + primal[:, i]**2 + tangent[:, i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, jnp.zeros(walkers.shape[0]))

    return _lapl_over_f


def generate_real_lattice(real_basis, real_cut):
    l0 = jnp.linalg.norm(real_basis, axis=-1).mean()  # get the mean length of the basis vectors
    
    img_range = jnp.arange(-2*real_cut, 2*real_cut+1)  # x2 to create sphere
    img_sets = list(product(*[img_range, img_range, img_range]))
    # first axis is the number of lattice vectors, second is the integers to scale the primitive vectors, third is the resulting set of vectors
    # then sum over those
    # print(len(img_sets))
    img_sets = jnp.concatenate([jnp.array(x)[None, :, None] for x in img_sets if not jnp.sum(jnp.array(x) == 0) == 3], axis=0)
    # print(img_sets.shape)
    imgs = jnp.sum(img_sets * real_basis, axis=1)

    # if a sphere around the image is within rcut then keep it
    lengths = jnp.linalg.norm(imgs, axis=-1)

    img_sets = jnp.any(lengths < (real_cut * l0), axis=1)
    imgs = imgs[mask]
    return imgs


def generate_reciprocal_lattice(reciprocal_basis, reciprocal_cut):
    # 3D uniform grids
    img_range = jnp.arange(-2*reciprocal_cut, 2*reciprocal_cut+1)
    img_sets = list(product(*[img_range, img_range, img_range]))  # cartesian product another worse version of this is available in cartesian_prod(...)

    img_sets = jnp.array([x for x in img_sets if not jnp.sum(x == 0) == 3])  # filter the zero vector
    reciprocal_lattice = jnp.dot(img_sets, reciprocal_basis)
    return reciprocal_lattice

def create_potential_energy_min_im(mol):
    """

    Notes:
        - May need to shift the origin to the center to enforce the spherical sum condition
        - I am now returning to length of unit cell units which is different to the unit cell length I was using before. How does this affect the computation?
        - Is the reciprocal height computed in the correct way?
    """

    def compute_pp_vectors_periodic(walkers):
        unit_cell_walkers = walkers.dot(mol.inv_real_basis)  # translate to the unit cell
        # tmp = jnp.bitwise_and(unit_cell_walkers > 1., unit_cell_walkers < 0.)
        # assert not jnp.any(tmp)  # does not work with tracing because it is input dependent
        re1 = jnp.expand_dims(unit_cell_walkers, axis=1)
        re2 = jnp.transpose(re1, [1, 0, 2])
        unit_cell_ee_vectors = re1 - re2
        min_image_unit_cell_ee_vectors = unit_cell_ee_vectors - (2 * unit_cell_ee_vectors).astype(int) * 1.  # 1 is length of unit cell put it here for clarity
        min_image_ee_vectors = min_image_unit_cell_ee_vectors.dot(mol.real_basis)
        return min_image_ee_vectors

    if mol.periodic_boundaries:
        
        real_basis = mol.real_basis
        reciprocal_basis = mol.reciprocal_basis
        kappa = mol.kappa
        volume = mol.volume

        real_lattice = generate_lattice(real_basis, mol.real_cut)  # (n_lattice, 3)
        reciprocal_lattice = generate_lattice(reciprocal_basis, mol.reciprocal_cut)
        rl_inner_product = inner(reciprocal_lattice, reciprocal_lattice)
        rl_factor = (4*jnp.pi / volume) * jnp.exp(- rl_inner_product / (4*kappa**2)) / rl_inner_product  

        e_charges = jnp.array([-1. for i in range(mol.n_el)])
        charges = jnp.concatenate([mol.z_atoms, e_charges], axis=0)  # (n_particle, )
        q_q = charges[None, :] * charges[:, None]  # q_i * q_j  (n_particle, n_particle)

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

            # put the walkers and r_atoms together
            walkers = jnp.concatenate([r_atoms, walkers], axis=0)  # (n_particle, 3)

            # compute the Rs0 term
            p_p_vectors = compute_pp_vectors_periodic(walkers)  # (n_particle, n_particle, 3)
            p_p_distances = jnp.linalg.norm(p_p_vectors, axis=-1)
            # p_p_distances[p_p_distances < 1e-16] = 1e200  # doesn't matter, diagonal dropped, this is just here to suppress the error
            Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # is half the value

            # compute the Rs > 0 term
            ex_vectors = p_p_vectors[..., None, :] - real_lattice[None, None, ...]  # (n_particle, n_particle, n_lattice, 3)
            ex_distances = jnp.linalg.norm(ex_vectors, axis=-1)
            Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
            real_sum = (q_q * (Rs0 + 0.5 * Rs1)).sum((-1, -2))
            
            # compute the constant factor
            self_interaction = - 0.5 * jnp.diag(q_q * 2 * kappa / jnp.sqrt(jnp.pi)).sum()
            constant = - 0.5 * charges.sum()**2 * jnp.pi / (kappa**2 * volume)  # is zero in neutral case

            # compute the reciprocal term reuse the ee vectors
            exp = jnp.real(jnp.sum(rl_factor[None, None, :] * jnp.exp(1j * p_p_vectors @ jnp.transpose(reciprocal_lattice)), axis=-1))
            reciprocal_sum = 0.5 * (q_q * exp).sum((-1,-2))
            
            potential = real_sum + reciprocal_sum + constant + self_interaction
            return potential

        return vmap(compute_potential_energy_solid_i, in_axes=(0, None, None))

    return vmap(compute_potential_energy_i, in_axes=(0, None, None))