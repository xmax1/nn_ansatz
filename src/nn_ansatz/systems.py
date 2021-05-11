
import numpy as np
import jax.numpy as jnp
import math
from pyscf import dft, gto
from pyscf.scf import RHF
import jax.random as rnd


def create_atom(r_atoms, z_atom):
    return [[int(charge), tuple(coord)] for charge, coord in zip(z_atom, r_atoms)]


def compute_reciprocal_basis(real_basis, volume):
    cv1, cv2, cv3 = real_basis.split(3, axis=0)
    rv1 = 2 * np.pi * np.cross(cv2.squeeze(), cv3.squeeze()) / volume
    rv2 = 2 * np.pi * np.cross(cv3.squeeze(), cv1.squeeze()) / volume
    rv3 = 2 * np.pi * np.cross(cv1.squeeze(), cv2.squeeze()) / volume
    reciprocal_basis = np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)
    return reciprocal_basis


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
                 r_atoms=None,
                 z_atoms=None,
                 n_el=None,
                 real_basis=None,
                 periodic_boundaries=False,
                 unit_cell_length=None,
                 real_cut=None,
                 reciprocal_cut=None,
                 kappa=None,
                 n_layers=2,
                 n_sh=64,
                 n_ph=16,
                 n_det=2,
                 step_size=0.05,
                 n_el_atoms=None,
                 n_up=None,
                 basis='sto3g',
                 device='cpu',
                 dtype=jnp.float32,
                 **kwargs):
        self.device, self.dtype = device, dtype

        if n_up is None:
            n_up = math.ceil(n_el / 2.)
        n_down = n_el - n_up

        self.n_el = n_el
        self.n_pairwise = n_el**2 - n_el
        self.n_up = n_up
        self.n_down = n_down
        self.spin = n_up - n_down  # will this be different for molecules?

        self.n_atoms = r_atoms.shape[0]
        self.atom = create_atom(r_atoms, z_atoms)
        self.charge = int(jnp.sum(z_atoms)) - n_el
        self.z_atoms = z_atoms
        self.r_atoms = r_atoms

        if n_el_atoms is None:
            n_el_atoms = jnp.array([int(x) for x in z_atoms])
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

        # ansatz
        self.n_layers = n_layers
        self.n_sh = n_sh
        self.n_ph = n_ph
        self.n_det = n_det

        # sampling
        self.step_size = step_size
        self.periodic_boundaries = periodic_boundaries
        if periodic_boundaries:
            
            self.real_basis = unit_cell_length * real_basis  # each basis vector is (1, 3)
            self.inv_real_basis = jnp.linalg.pinv(self.real_basis)
            self.r_atoms = self.r_atoms * unit_cell_length
            self.volume = compute_volume(self.real_basis)
            self.reciprocal_basis = compute_reciprocal_basis(self.real_basis, self.volume)
            self.reciprocal_height = compute_reciprocal_height(self.reciprocal_basis, unit_cell_length)
            self.unit_cell_length = unit_cell_length

            self.real_cut = real_cut * unit_cell_length
            self.reciprocal_cut = reciprocal_cut
            self.kappa = kappa


            print('Cell: \n',
              'real_basis:', '\n', self.real_basis, '\n',
              'reciprocal_basis:', '\n', self.reciprocal_basis, '\n',
              'real_cut         = %.2f \n' % self.real_cut,
              'reciprocal_cut   = %i \n' % self.reciprocal_cut,
              'kappa            = %.2f \n' % kappa,
              'volume           = %.2f \n' % self.volume)



            



    @property
    def atom_positions(self):
        return [x for x in self.r_atoms.split(self.n_atoms, axis=0)]

    def initialise_walkers(self, params=None, d0s=None, equilibrate=None, walkers=None, n_walkers: int=1024, n_it=64):
        if walkers is None:
            return initialise_walkers(self.n_el_atoms, self.atom_positions, n_walkers)
        if len(walkers) == n_walkers:
            return walkers
        if params is None or d0s is None or equilibrate is None:
            exit('If the number of loaded walkers is not equal to the number of requested walkers \n'
                 'params, d0s and equilibrate function must be passed to initialize walkers function for equilibration')
        n_replicate = math.ceil(n_walkers / len(walkers))
        walkers = jnp.concatenate([walkers for i in range(n_replicate)], axis=0)
        walkers = walkers[:n_walkers, ...]
        key = rnd.PRNGKey(0)  # this may be an issue if we double the batch size and then double the available gpus.
        walkers = equilibrate(params, walkers, d0s, key, n_it=n_it)
        return walkers



def initialise_walkers(ne_atoms, atom_positions, n_walkers):
    r""" Initialises walkers for pretraining

        Usage:
            walkers = initialize_walkers(ne_atoms, atom_positions, n_walkers).to(device=self.device, dtype=self.dtype)

        Args:
            ne_atoms (list int): number of electrons assigned to each nucleus
            atom_positions (list np.array): positions of the nuclei
            n_walkers (int): number of walkers

        Returns:
            walkers (np.array): walker positions (n_walkers, n_el, 3)

        """
    key = rnd.PRNGKey(1)
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            key, subkey = rnd.split(key)

            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = rnd.normal(subkey, (n_walkers, 1, 3)) + atom_position
                ups.append(curr_sample_up)
            else:
                curr_sample_down = rnd.normal(subkey, (n_walkers, 1, 3)) + atom_position
                downs.append(curr_sample_down)

    ups = jnp.concatenate(ups, axis=1)
    downs = jnp.concatenate(downs, axis=1)
    curr_sample = jnp.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return curr_sample

