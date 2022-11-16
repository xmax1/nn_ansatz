
import numpy as np
import jax.numpy as jnp
import math
from pyscf import gto
from pyscf.scf import RHF
from itertools import product, combinations
from .utils import split_variables_for_pmap
from .ansatz_base import transform_vector_space


def create_atom(r_atoms, z_atom):
    return [[int(charge), tuple(coord)] for charge, coord in zip(z_atom, r_atoms)]


def compute_reciprocal_basis(basis, volume):
    cv1, cv2, cv3 = basis.split(3, axis=0)
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

def create_system_ansatz(cfg):
    if cfg['pbc']:
        return PBCAnsatz(**cfg)
    else:
        return IsolatedAnsatz(**cfg)


class PBCAnsatz():
    def __init__(self, ):
        return

class IsolatedAnsatz():
    def __init__(self, ):
        return 


def add_nfuncs(func_split, n_funcs):
    for f in func_split:
        if any(char.isdigit() for char in f):
            n_funcs += (int(f[:-3]) - 1)
    return n_funcs


def get_nfunc_in(func):
    split = func.split('+')
    n_funcs = 0
    if 'bowl' in func:
        n_funcs += 1
    if 'sin' in func:
        sin_desc = [x for x in split if 'sin' in x][0]
        nsin = int(sin_desc[:-3]) if len(sin_desc) > 3 else 1
        n_funcs += nsin
    if 'cos' in func:
        cos_desc = [x for x in split if 'cos' in x][0]
        ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1
        n_funcs += ncos
    return n_funcs


class SystemAnsatz():
    def __init__(self,
                 system=None,
                 r_atoms=None,
                 z_atoms=None,
                 n_el=None,
                 n_el_atoms=None,
                 n_up=None,
                 basis=None,
                 kpoints=None,
                 inv_basis=None,
                 pbc=False,
                 spin_polarized=False,
                 density_parameter=None,
                 kappa=None,
                 simulation_cell: tuple = (1, 1, 1),
                 orbitals='anisotropic',
                 einsum:bool = False,
                 nonlinearity: str = 'tanh',
                 input_activation_nonlinearity: str = 'sin',
                 jastrow: bool = False,
                 backflow_coords: bool = True,
                 bf_af: str='no_af',
                 psplit_spins: bool = True,
                 n_layers=2,
                 n_sh=64,
                 n_ph=16,
                 n_det=2,
                 step_size=0.02,
                 correlation_length=10,
                 n_walkers=256,   
                 orbital_basis='sto3g',
                 device='cpu',
                 dtype=jnp.float32,
                 atoms_from_unit_cell=True,
                 scale_cell=1.,
                 print_ansatz=True,
                 atol=1e-5,
                 target_acceptance=0.5,
                 **kwargs):

        self.system = system
        self.jastrow = jastrow
        self.psplit_spins = psplit_spins
        self.target_acceptance = target_acceptance

        self.device, self.dtype = device, dtype
        self.n_walkers_per_device = kwargs['n_walkers_per_device']
        self.n_devices = kwargs['n_devices']
        self.bf_af = bf_af

        # ELECTRONS
        if simulation_cell is not None: n_el = int(n_el * jnp.prod(jnp.array(simulation_cell)))
        if n_up is None: n_up = math.ceil(n_el / 2.) if not spin_polarized else n_el

        self.n_el = n_el
        self.n_pairwise = n_el**2 - n_el
        self.n_up = n_up
        self.n_down = n_el - n_up
        self.spin = n_up - self.n_down  # will this be different for molecules?

        # SYSTEM SIZE
        if system == 'HEG' and density_parameter is not None:
            self.density_parameter = density_parameter
            v_per_electron = 4. * jnp.pi / 3.
            volume = n_el * v_per_electron
            scale_cell = volume**(1./3.)

        if pbc:
            self.kappa = kappa

            single_cell_basis = basis * scale_cell
            simulation_cell_transforms = [jnp.array([x, y, z])[:, None] for z in range(simulation_cell[2]) for y in range(simulation_cell[1]) for x in range(simulation_cell[0])]
            simulation_cell = jnp.array(simulation_cell)[:, None]
            basis = single_cell_basis * simulation_cell

            inv_basis = jnp.linalg.inv(basis)
            volume = compute_volume(basis)

            self.inv_basis = jnp.diag(inv_basis)[None, :] if jnp.all(inv_basis.sum(0) == jnp.diag(inv_basis)) else inv_basis
            self.basis = jnp.diag(basis)[None, :] if jnp.all(basis.sum(0) == jnp.diag(basis)) else basis
            self.reciprocal_basis = compute_reciprocal_basis(basis, volume)
            self.scale_cell = scale_cell
            self.volume = volume
            self.atol = atol # tolerance for converging the ewalds sum

            if r_atoms is not None:
                if atoms_from_unit_cell: 
                    r_atoms = transform_vector_space(r_atoms, single_cell_basis)
                r_atoms = jnp.concatenate([r_atoms + (single_cell_basis * sct).sum(0) for sct in simulation_cell_transforms], axis=0)
                n_cells = len(simulation_cell_transforms)
                z_atoms = z_atoms[None, :].repeat(n_cells, axis=0).reshape(-1)
                n_el_atoms = [item for sublist in [n_el_atoms for _ in range(n_cells)] for item in sublist]
                assert sum(n_el_atoms) == n_el

            if print_ansatz:
                print('Cell: \n',
                'pbc', pbc, '\n',
                'basis:', '\n', self.basis, '\n',
                'inv_basis:', '\n', self.inv_basis, '\n',
                'scale_cell: ', '\n', self.scale_cell, '\n',
                'reciprocal_basis:', '\n', self.reciprocal_basis, '\n',
                'kappa            = %.2f \n' % self.kappa, '\n',
                'volume           = %.2f \n' % self.volume)

            shells = [1, 7, 19, 27, 33, 57]
            # shell = shells.index(n_el) + 1 + 2 # + 1 for correct the index + 2 to take the next shell up for partial polarization
            shell = 6
            kpoints = generate_k_points(n_shells=shell) * 2 * jnp.pi
            kpoints = transform_vector_space(kpoints, inv_basis)

        self.pbc = pbc
        self.basis = basis
        self.inv_basis = inv_basis 
        self.kpoints = kpoints
        self.backflow_coords = backflow_coords

        # ATOMS
        self.n_atoms = r_atoms.shape[0] if r_atoms is not None else 0
        self.charge = int(jnp.sum(z_atoms)) - n_el if r_atoms is not None else - n_el
        self.z_atoms = z_atoms
        self.r_atoms = r_atoms

        self.n_el_atoms = n_el_atoms

        if print_ansatz:
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

        nfunc_in = get_nfunc_in(input_activation_nonlinearity)
        self.n_in = 4 if not pbc else 3 * nfunc_in
        self.n_sh_in = self.n_in * self.n_atoms if not self.n_atoms == 0 else self.n_in
        if 'kpoints' in input_activation_nonlinearity: 
            self.n_sh_in += int([x for x in input_activation_nonlinearity.split('+') if 'kpoints' in x][0][:-7]) - 1
        self.n_ph_in = self.n_in + 1
        self.orbitals = orbitals
        self.einsum = einsum
        self.nonlinearity = nonlinearity
        self.input_activation_nonlinearity = input_activation_nonlinearity

        # sampling
        self.step_size = split_variables_for_pmap(self.n_devices, step_size)
        self.correlation_length = correlation_length
        self.n_walkers = n_walkers

        if not pbc:
            self.atom = create_atom(r_atoms, z_atoms)

            mol = gto.Mole(
                atom=self.atom,
                unit='Bohr',
                basis=orbital_basis,
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
        return self.r_atoms.split(self.n_atoms, axis=0) if self.r_atoms is not None else None


def generate_k_shells(n_shells):
    img_range = jnp.arange(-3, 3+1)  # preset, when are we ever going to use more
    img_sets = jnp.array(list(product(*[img_range, img_range, img_range])))
    norms = jnp.linalg.norm(img_sets, axis=-1)
    idxs = jnp.argsort(norms)
    img_sets, norms = img_sets[idxs], norms[idxs]
    norm = 0.
    k_shells = {norm: [jnp.array([0.0, 0.0, 0.0])]}  # leacing the dictionary logic in case we ever need this data structure
    for k_point, norm_tmp in zip(img_sets[1:], norms[1:]):
        if norm_tmp > norm:
            if len(k_shells) == n_shells:
                break
            norm = norm_tmp
            k_shells[float(norm)] = [k_point]
        else:
            if np.any([(k_point == x).all() for x in k_shells[float(norm)]]):
                continue # because we include the opposite k_point in the sequence this statement avoids repeats
            k_shells[float(norm)].append(k_point)
        k_shells[float(norm)].append(-k_point)
    return k_shells


def generate_k_points(n_shells):
    k_shells = generate_k_shells(n_shells)
    k_points = []
    for k, v in k_shells.items():
        for k_point in v:
            k_points.append(k_point)
    return jnp.array(k_points)

    
def generate_plane_vectors(primitives, origin, contra, center):
    origin_pairs = list(combinations(primitives, 2))
    origin_pairs = [np.squeeze(x) for x in np.split(np.array(origin_pairs), 2, axis=1)]
    origin_plane_vectors = np.cross(origin_pairs[0], origin_pairs[1])
    
    tmp = np.stack([origin_plane_vectors, -origin_plane_vectors], axis=1)  # put the cross product in either direction
    origin_plane_vectors = use_vector_facing_away(origin, center, tmp)  # select the direction that points away from the center

    contra_pairs = list(combinations(-primitives, 2))
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
