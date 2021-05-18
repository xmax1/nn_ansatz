
import numpy as np

import pyscf
from pyscf.pbc.gto.cell import ewald

from pyscf.pbc import gto, scf, grad
from pyscf import lib

def compute_volume(v1, v2, v3):
    cross = np.cross(v2, v3, axisa=0, axisb=0)
    box = np.sum(v1 * cross)
    return np.abs(np.squeeze(box))

#https://unlcms.unl.edu/cas/physics/tsymbal/teaching/SSP-927/Section%2001_Crystal%20Structure.pdf

basis_cell = np.array([[-0.5, 0.5, 0.5],
                       [0.5, -0.5, 0.5],
                       [0.5, 0.5, -0.5]])
print(abs(np.linalg.det(basis_cell)))
inv_cell_basis = np.linalg.inv(basis_cell)
center = np.sum(basis_cell, axis=0) / 2.
print('center: ', center)

cv1, cv2, cv3 = np.split(basis_cell, 3, axis=1)
print('primitive vectors: ', '\n', cv1, '\n', cv2, '\n', cv3)
volume = compute_volume(cv1, cv2, cv3)
print('volume', volume)
vector_set = [cv1, -cv1, cv2, -cv2, cv3, -cv3]

rv1 = 2 * np.pi * np.cross(cv2.squeeze(), cv3.squeeze()) / volume
rv2 = 2 * np.pi * np.cross(cv3.squeeze(), cv1.squeeze()) / volume
rv3 = 2 * np.pi * np.cross(cv1.squeeze(), cv2.squeeze()) / volume

reciprocal_volume = abs(np.linalg.det(np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)))
print('reciprocal volume: ', reciprocal_volume)

print('recipricol vectors: ', '\n', rv1, '\n', rv2, '\n', rv3)
vector_set_recip = [x.transpose() for x in (rv1, -rv1, rv2, -rv2, rv3, -rv3)]

n_walkers = 1
n_el = 4

walkers = np.random.uniform(0, 0.5, (n_walkers, n_el, 3)) - center
talkers = walkers.dot(inv_cell_basis)
talkers = np.fmod(talkers, 1.)
talkers = np.where(talkers < 0, talkers + 1., talkers)
walkers = talkers.dot(basis_cell)
e_charges = np.array([[-1.]]).repeat(n_walkers, axis=0).repeat(n_el, axis=1)

r_atoms = np.array([[[0.25, 0.25, 0.25]]]).repeat(n_walkers, axis=0) - center
print('r_atom: ', r_atoms)

r_charges = np.array([[float(n_el)]]).repeat(n_walkers, axis=0)

walkers = np.concatenate([r_atoms, walkers], axis=1)
charges = np.concatenate([r_charges, e_charges], axis=1)
print(walkers.shape, charges.shape)



cell = gto.Cell()
cell.atom= [['Li', [0.25, 0.25, 0.25]]]
basis_cell = np.array([[-0.5, 0.5, 0.5],
                       [0.5, -0.5, 0.5],
                       [0.5, 0.5, -0.5]])
cell.a = basis_cell
cell.unit = 'bohr'
cell.spin = 1.
cell.build()

# ew_eta, ew_cut = cell.get_ewald_params()
# print('eta', 'cut', cell.get_ewald_params())
#
# print(len(cell.get_lattice_Ls(rcut=ew_cut)))
#
# print(cell.dimension)
#
# b = cell.reciprocal_vectors(norm_to=1)
# heights_inv = lib.norm(b, axis=1)
# print(b)
# print(heights_inv)
# # print(ewald(cell, ew_eta=2., ew_cut=2.))
#
# # print(cell.get_lattice_Ls(rcut=10.))
#
# print(len(cell.get_lattice_Ls(rcut=ew_cut)))
#
# print(cell.a)
#
# print(cell.mesh)
print(ewald(cell))
print('precision: ', cell.precision)
print('nbas: ', cell.nbas, cell.bas_angular(0), cell.bas_exp(0))

