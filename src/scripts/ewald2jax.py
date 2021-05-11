import numpy as np
from itertools import chain, combinations, combinations_with_replacement, product
# from scipy.special import erfc
import pickle as pk
from time import time
import math
from jax import vmap
from jax.scipy.special import erfc

import jax.numpy as jnp
from nn_ansatx import setup, SystemAnsatz

def vector_sub(v1, v2, axis=-2):
    return jnp.expand_dims(v1, axis=axis) - jnp.expand_dims(v2, axis=axis-1)

def vector_add(v1, v2, axis=-2):
    return jnp.expand_dims(v1, axis=axis) + jnp.expand_dims(v2, axis=axis-1)

def compute_distances(v1, v2):
    inter_vector = vector_sub(v1, v2)
    return jnp.sqrt(jnp.sum(inter_vector**2, axis=-1))

def inner(v1, v2):
    return jnp.sum(v1 * v2, axis=-1)

def pbc(walkers, real_basis, inv_real_basis):
    talkers = walkers.dot(inv_real_basis)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0, talkers + 1., talkers)
    walkers = talkers.dot(real_basis)
    return walkers


def generate_real_lattice(real_basis, rcut, reciprocal_height):
    # from pyscf, some logic to set the number of imgs away from simulation cell. Adapted version confirmed in notion
    nimgs = jnp.ceil(rcut*reciprocal_height + 1.1).astype(int)
    img_range = jnp.arange(-nimgs, nimgs+1)
    img_sets = list(product(*[img_range, img_range, img_range]))
    # first axis is the number of lattice vectors, second is the integers to scale the primitive vectors, third is the resulting set of vectors
    # then sum over those
    # print(len(img_sets))
    img_sets = jnp.concatenate([jnp.array(x)[None, :, None] for x in img_sets if not jnp.sum(jnp.array(x) == 0) == 3], axis=0)
    # print(img_sets.shape)
    imgs = jnp.sum(img_sets * real_basis, axis=1)
    
    # generate all the single combinations of the basis vectors
    v = jnp.split(real_basis, 3, axis=0)
    z = jnp.zeros_like(v[0])
    vecs = product(*[[-v[0], z, v[0]],[-v[1], z, v[1]], [-v[2], z, v[2]]])
    vecs = jnp.array(list(vecs)).squeeze().sum(-2)  # sphere around the origin

    # if a sphere around the image is within rcut then keep it
    lengths = jnp.linalg.norm(vecs[None, ...] + imgs[:, None, :], axis=-1)
    mask = jnp.any(lengths < rcut, axis=1)
    nimgs = len(imgs)
    imgs = imgs[mask]
    return imgs


def generate_reciprocal_lattice(reciprocal_basis, mesh):
    # 3D uniform grids
    rx = jnp.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = jnp.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = jnp.fft.fftfreq(mesh[2], 1./mesh[2])
    base = (rx, ry, rz)
    cartesian_product = jnp.array(list(product(*base)))  # another worse version of this is available in cartesian_prod(...)
    cartesian_product = jnp.array([x for x in cartesian_product if not jnp.sum(x == 0) == 3])  # filter the zero vector
    reciprocal_lattice = jnp.dot(cartesian_product, reciprocal_basis)
    return reciprocal_lattice


def compute_volume(basis):
    v1, v2, v3 = basis.split(3, axis=0)
    cross = jnp.cross(v2, v3, axisa=1, axisb=1)
    box = jnp.sum(v1 * cross)
    return jnp.abs(np.squeeze(box))


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
        mesh = [mol.reciprocal_cut for i in range(3)]
        volume = compute_volume(real_basis)

        real_lattice = generate_real_lattice(real_basis, mol.real_cut, mol.reciprocal_height)  # (n_lattice, 3)
        reciprocal_lattice = generate_reciprocal_lattice(reciprocal_basis, mesh)
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
            p_p_vectors = vector_sub(walkers, walkers)
            p_p_distances = compute_distances(walkers, walkers)
            # p_p_distances[p_p_distances < 1e-16] = 1e200  # doesn't matter, diagonal dropped, this is just here to suppress the error
            Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # is half the value

            # compute the Rs > 0 term
            ex_walkers = vector_add(walkers, real_lattice)  # (n_particle, n_lattice, 3)
            tmp = walkers[:, None, None, :] - ex_walkers[None, ...]  # (n_particle, n_particle, n_lattice, 3)
            ex_distances = jnp.sqrt(jnp.sum(tmp**2, axis=-1))  
            Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
            real_sum = 0.5 * (q_q * (Rs0 + Rs1)).sum((-1, -2))
            
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


def batched_cdist_l2(x1, x2):

    x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True)
    x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True)
    cdist = jnp.sqrt(jnp.swapaxes(x1_sq, -1, -2) + x2_sq \
                     - jnp.sum(2 * jnp.expand_dims(x1, axis=0) * jnp.expand_dims(x2, axis=1), axis=-1))
    return cdist


#https://unlcms.unl.edu/cas/physics/tsymbal/teaching/SSP-927/Section%2001_Crystal%20Structure.pdf

# REAL LATTICE
real_basis = np.array([[-0.5, 0.5, 0.5],
                       [0.5, -0.5, 0.5],
                       [0.5, 0.5, -0.5]])
inv_real_basis = np.linalg.inv(real_basis)

# SYSTEM
n_walkers = 1
n_el = 3

config = setup(system='LiSolid',
               kappa=1.,
               real_cut=1.,
               reciprocal_cut=3)

mol = SystemAnsatz(**config)

walkers = pbc(np.random.uniform(0, 0.5, (n_walkers, n_el, 3)) * mol.unit_cell_length, real_basis, inv_real_basis)

pe = create_potential_energy(mol)

print(pe(walkers, mol.r_atoms, mol.z_atoms))

walkers = walkers[0]

real_basis = mol.real_basis
reciprocal_basis = mol.reciprocal_basis
kappa = mol.kappa
mesh = [mol.reciprocal_cut for i in range(3)]
volume = compute_volume(real_basis)

real_lattice = generate_real_lattice(real_basis, mol.real_cut, mol.reciprocal_height)  # (n_lattice, 3)
reciprocal_lattice = generate_reciprocal_lattice(reciprocal_basis, mesh)
rl_inner_product = inner(reciprocal_lattice, reciprocal_lattice)
rl_factor = (4*jnp.pi / volume) * jnp.exp(- rl_inner_product / (4*kappa**2)) / rl_inner_product  

e_charges = jnp.array([-1. for i in range(mol.n_el)])
charges = jnp.concatenate([mol.z_atoms, e_charges], axis=0)  # (n_particle, )
q_q = charges[None, :] * charges[:, None]  # q_i * q_j  (n_particle, n_particle)


# put the walkers and r_atoms together
walkers = jnp.concatenate([mol.r_atoms, walkers], axis=0)  # (n_particle, 3)

# compute the Rs0 term
p_p_vectors = vector_sub(walkers, walkers)
p_p_distances = compute_distances(walkers, walkers)
# p_p_distances[p_p_distances < 1e-16] = 1e200  # doesn't matter, diagonal dropped, this is just here to suppress the error
Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # is half the value

# compute the Rs > 0 term
ex_walkers = vector_add(walkers, real_lattice)  # (n_particle, n_lattice, 3)
tmp = walkers[:, None, None, :] - ex_walkers[None, ...]  # (n_particle, n_particle, n_lattice, 3)
ex_distances = jnp.sqrt(jnp.sum(tmp**2, axis=-1))  
Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
real_sum = 0.5 * (q_q * (Rs0 + Rs1)).sum((-1, -2))

# compute the constant factor
self_interaction = - 0.5 * jnp.diag(q_q * 2 * kappa / jnp.sqrt(jnp.pi)).sum()
constant = - 0.5 * charges.sum()**2 * jnp.pi / (kappa**2 * volume)  # is zero in neutral case

# compute the reciprocal term reuse the ee vectors
exp = jnp.real(jnp.sum(rl_factor[None, None, :] * jnp.exp(1j * p_p_vectors @ jnp.transpose(reciprocal_lattice)), axis=-1))
reciprocal_sum = 0.5 * (q_q * exp).sum((-1,-2))

potential = real_sum + reciprocal_sum + constant + self_interaction

print(potential)
# #https://unlcms.unl.edu/cas/physics/tsymbal/teaching/SSP-927/Section%2001_Crystal%20Structure.pdf

# # REAL LATTICE

# inv_real_basis = np.linalg.inv(real_basis)
# center = np.sum(real_basis, axis=0) / 2.
# cv1, cv2, cv3 = np.split(real_basis, 3, axis=1)
# volume = compute_volume(cv1, cv2, cv3)

# # RECIPROCAL LATTICE
# rv1 = 2 * np.pi * np.cross(cv2.squeeze(), cv3.squeeze()) / volume
# rv2 = 2 * np.pi * np.cross(cv3.squeeze(), cv1.squeeze()) / volume
# rv3 = 2 * np.pi * np.cross(cv1.squeeze(), cv2.squeeze()) / volume
# reciprocal_basis = np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)
# normed_to1 = reciprocal_basis / (2 * np.pi)
# reciprocal_height = np.linalg.norm(normed_to1, axis=1)
# reciprocal_volume = abs(np.linalg.det(reciprocal_basis))

# # SYSTEM
# n_walkers = 1
# n_el = 3

# walkers = pbc(np.random.uniform(0, 0.25, (n_walkers, n_el, 3)) - center)

# r_atoms = np.array([[[0.25, 0.25, 0.25]]]).repeat(n_walkers, axis=0) - center
# r_charges = np.array([[float(n_el)]]).repeat(n_walkers, axis=0)
# e_charges = np.array([[-1.]]).repeat(n_walkers, axis=0).repeat(n_el, axis=1)

# charges = np.concatenate([r_charges, e_charges], axis=1)

# # THINGS THAT WE NEED THAT ARE FIXED FOR NOW
# mesh = [10, 10, 10]  # from cell.mesh
# precision = 1e-8 # from cell.precision


# # DETAILS
# print('center: ', center)
# print('primitive vectors: ', '\n', cv1, '\n', cv2, '\n', cv3)
# print('volume: ', abs(np.linalg.det(real_basis)))
# # print('r_atom: ', r_atoms)
# print('reciprocal volume: ', reciprocal_volume)
# print('recipricol vectors: ', '\n', rv1, '\n', rv2, '\n', rv3)
# print('mesh: ', mesh)
# print('precision: ', precision)