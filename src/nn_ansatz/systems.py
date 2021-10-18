
import numpy as np
import jax.numpy as jnp
import math
from pyscf import dft, gto
from pyscf.scf import RHF
import jax.random as rnd
import itertools
from .utils import key_gen, split_variables_for_pmap
from jax import pmap


def create_atom(r_atoms, z_atom):
    return [[int(charge), tuple(coord)] for charge, coord in zip(z_atom, r_atoms)]


def compute_reciprocal_basis(real_basis, volume):
    cv1, cv2, cv3 = real_basis.split(3, axis=0)
    rv1 = np.cross(cv2.squeeze(), cv3.squeeze()) / volume
    rv2 = np.cross(cv3.squeeze(), cv1.squeeze()) / volume
    rv3 = np.cross(cv1.squeeze(), cv2.squeeze()) / volume
    reciprocal_basis = np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)
    return reciprocal_basis * 2 * np.pi


def compute_reciprocal_height(reciprocal_basis, unit_cell_length):
    normed_to1 = reciprocal_basis / (2 * np.pi * unit_cell_length)  # not sure what this is doing
    reciprocal_height = np.linalg.norm(normed_to1, axis=1)[0]
    return reciprocal_height


def compute_volume(basis):
    v1, v2, v3 = basis.split(3, axis=0)
    cross = jnp.cross(v2, v3, axisa=1, axisb=1)
    box = jnp.sum(v1 * cross)
    return jnp.abs(jnp.squeeze(box))


class SystemAnsatz():
    def __init__(self,
                 system=None,
                 r_atoms=None,
                 z_atoms=None,
                 n_el=None,
                 real_basis=None,
                 periodic_boundaries=False,
                 unit_cell_length=None,
                 real_cut=None,
                 reciprocal_cut=None,
                 kappa=None,
                 scalar_inputs=False,
                 orbitals='anisotropic',
                 n_periodic_input=1,
                 n_layers=2,
                 n_sh=64,
                 n_ph=16,
                 n_det=2,
                 step_size=0.05,
                 correlation_length=10,
                 n_walkers=256,
                 n_el_atoms=None,
                 n_up=None,
                 basis='sto3g',
                 device='cpu',
                 dtype=jnp.float32,
                 **kwargs):

        self.system = system

        self.device, self.dtype = device, dtype
        self.n_walkers_per_device = kwargs['n_walkers_per_device']
        self.n_devices = kwargs['n_devices']

        if n_up is None:
            n_up = math.ceil(n_el / 2.)
        n_down = n_el - n_up

        self.n_el = n_el
        self.n_pairwise = n_el**2 - n_el
        self.n_up = n_up
        self.n_down = n_down
        self.spin = n_up - n_down  # will this be different for molecules?

        self.n_atoms = r_atoms.shape[0]
        self.charge = int(jnp.sum(z_atoms)) - n_el
        self.z_atoms = z_atoms
        self.r_atoms = r_atoms

        if n_el_atoms is None: n_el_atoms = jnp.array([int(x) for x in z_atoms])
        self.n_el_atoms = n_el_atoms

        print('System: \n',
              'n_atoms = %i \n' % self.n_atoms,
              'n_up    = %i \n' % self.n_up,
              'n_down  = %i \n' % self.n_down,
              'n_el    = %i \n' % self.n_el)

        print('Ansatz: \n',
              'n_layers = %i \n' % n_layers,
              'n_det    = %i \n' % n_det,
              'n_sh     = %i \n' % n_sh,
              'n_ph     = %i \n' % n_ph)

        # ansatz
        self.n_layers = n_layers
        self.n_sh = n_sh
        self.n_ph = n_ph
        self.n_det = n_det
        self.scalar_inputs = scalar_inputs
        self.n_in = 1 if scalar_inputs else 4
        self.orbitals=orbitals

        # throwaway
        self.min_cell_width = 1.

        # sampling
        self.step_size = split_variables_for_pmap(self.n_devices, step_size)
        self.correlation_length = correlation_length
        self.n_walkers = n_walkers
        self.periodic_boundaries = periodic_boundaries

        self.unit_cell_length = unit_cell_length  # has to be here for in the ansatz
        self.real_basis = unit_cell_length * real_basis if periodic_boundaries else None # each basis vector is (1, 3)
        self.inv_real_basis = jnp.linalg.inv(self.real_basis) if periodic_boundaries else None

        if periodic_boundaries:

            self.n_in = 1 if scalar_inputs else (3 * n_periodic_input) + 1
            self.n_periodic_input = n_periodic_input
            
            self.r_atoms = self.r_atoms * unit_cell_length
            self.volume = compute_volume(self.real_basis)
            self.reciprocal_basis = compute_reciprocal_basis(self.real_basis, self.volume)
            self.l0 = float(jnp.min(jnp.linalg.norm(self.real_basis, axis=-1)))
            self.min_cell_width = compute_min_width_of_cell(self.real_basis)

            self.real_cut = real_cut
            self.reciprocal_cut = reciprocal_cut
            self.kappa = kappa

            print('Cell: \n',
              'real_basis:', '\n', self.real_basis, '\n',
              'reciprocal_basis:', '\n', self.reciprocal_basis, '\n',
              'real_cut         = %.2f \n' % self.real_cut,
              'reciprocal_cut   = %i \n' % self.reciprocal_cut,
              'kappa            = %.2f \n' % kappa,
              'volume           = %.2f \n' % self.volume,
              'min_cell_width   = %.2f \n' % self.min_cell_width,
              'n_periodic_input = %i \n' % n_periodic_input,
              'unit_cell_length = %.2f \n' % self.unit_cell_length)

        self.atom = create_atom(r_atoms, z_atoms)

        mol = gto.Mole(
            atom=self.atom,
            unit='Bohr',
            basis=basis,
            charge=self.charge,
            spin=self.spin
        )
        mol.build()
        self.pyscf_mol = mol

        mf = RHF(mol)
        mf.kernel()

        self.mf = mf
        self.moT = jnp.array(mf.mo_coeff.T)

    @property
    def atom_positions(self):
        return [x for x in self.r_atoms.split(self.n_atoms, axis=0)]

    
def generate_plane_vectors(primitives, origin, contra, center):
    origin_pairs = list(itertools.combinations(primitives, 2))
    origin_pairs = [np.squeeze(x) for x in np.split(np.array(origin_pairs), 2, axis=1)]
    origin_plane_vectors = np.cross(origin_pairs[0], origin_pairs[1])
    
    tmp = np.stack([origin_plane_vectors, -origin_plane_vectors], axis=1)  # put the cross product in either direction
    origin_plane_vectors = use_vector_facing_away(origin, center, tmp)  # select the direction that points away from the center

    contra_pairs = list(itertools.combinations(-primitives, 2))
    contra_pairs = [np.squeeze(x) for x in np.split(np.array(contra_pairs), 2, axis=1)]
    contra_plane_vectors = np.cross(contra_pairs[0], contra_pairs[1])
    
    tmp = np.stack([contra_plane_vectors, -contra_plane_vectors], axis=1)
    contra_plane_vectors = use_vector_facing_away(contra, center, tmp)
    
    points = [origin for i in range(3)]
    [points.append(contra)  for i in range(3)]
    
    vectors = np.concatenate([origin_plane_vectors, contra_plane_vectors], axis=0)
    
    return points, vectors


def use_vector_facing_away(point0, center, vectors):
    # checks if the vector is closer or further from the center
    # this method assumes that given the origin or contra, moving along one of the 3 related plane vectors 
    new_point = point0 + vectors
    dists = np.sqrt(np.sum((new_point - center)**2, axis=-1))
    idxs = np.argmax(dists, axis=1)
    return np.stack([v[pair_idx, :] for (pair_idx, v) in zip(idxs, vectors)], axis=0)


def get_ds(points, vectors):
    # p0 is a point in the plane
    # ax + by + cz + d = 0
    # a, b, and c determined by the normal vector
    # this is independent of the sign of the normal vector (as expected)
    # what's ds? deez nuts
    # no but really it's the constant in the equation of a plane
    x = [np.sum(- pv * p0) for p0, pv in zip(points[:3], vectors[:3])]
    [x.append(np.sum(pv * p0)) for p0, pv in zip(points[3:], vectors[3:])] # change the sign because the normal is pointing in the wrong direction
    return x


def compute_distance(d0, d1, v):
    return np.abs(d0 - d1) / np.sqrt(np.sum(v**2))


def get_distances(ds, vectors):
    ds = np.split(np.array(ds), 2)
    vectors = np.split(np.array(vectors), 2, axis=0)
    return np.array([compute_distance(d0, d1, v) for d0, d1, v in zip(ds[0], ds[1], vectors[0])])


def compute_min_width_of_cell(basis):
    origin = np.array([0.0, 0.0, 0.0])
    contra = np.sum(basis, axis=0)  # the opposite point
    center = contra / 2.  
    points, vectors = generate_plane_vectors(basis, origin, contra, center)
    ds = get_ds(points, vectors)
    distances = get_distances(ds, vectors)
    return jnp.min(distances)
