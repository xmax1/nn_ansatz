
import numpy as np
import jax.numpy as jnp
import math
from pyscf import dft, gto
from pyscf.scf import RHF


def create_atom(r_atoms, z_atom):
    return [[int(charge), tuple(coord)] for charge, coord in zip(z_atom, r_atoms)]


class SystemAnsatz():
    def __init__(self,
                 r_atoms,
                 z_atoms,
                 n_el,
                 n_layers=2,
                 n_sh=64,
                 n_ph=16,
                 n_det=2,
                 step_size=0.05,
                 n_el_atoms=None,
                 n_up=None,
                 basis='sto3g',
                 device='cpu',
                 dtype=jnp.float32):
        self.device, self.dtype = device, dtype

        if n_up is None:
            n_up = math.ceil(n_el / 2.)
        n_down = n_el - n_up

        self.n_el = n_el
        self.n_up = n_up
        self.n_down = n_down
        self.spin = n_up - n_down  # will this be different for molecules?

        self.n_atoms = r_atoms.shape[0]
        self.atom = create_atom(r_atoms, z_atoms)
        self.charge = int(jnp.sum(z_atoms)) - n_el
        self.z_atoms = z_atoms
        self.r_atoms = r_atoms

        if n_el_atoms is None:
            self.n_el_atoms = [int(x) for x in z_atoms]

        print('System: \n',
              'n_atoms = %i \n' % self.n_atoms,
              'n_up    = %i \n' % n_up,
              'n_down  = %i \n' % n_down)

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

    @property
    def atom_positions(self):
        return [x for x in self.r_atoms.split(self.n_atoms, axis=0)]

    def initialise_walkers(self, n_walkers: int=1024):
        return initialize_walkers(self.n_el_atoms, self.atom_positions, n_walkers)


def initialize_walkers(ne_atoms, atom_positions, n_walkers):
    r""" Initialises walkers for pretraining

        Usage:
            walkers = initialize_walkers(ne_atoms, atom_positions, n_walkers).to(device=self.device, dtype=self.dtype)

        Args:
            ne_atoms (list int): number of electrons assigned to each nucleus
            atom_positions (list np.array): positions of the nuclei
            n_walkers (int): number of walkers

        Returns:
            walkers (np.array): walker positions (n_walkers, n_elec, 3)

        """
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = np.random.normal(loc=atom_position, scale=1., size=(n_walkers, 1, 3))
                ups.append(curr_sample_up)
            else:
                curr_sample_down = np.random.normal(loc=atom_position, scale=1., size=(n_walkers, 1, 3))
                downs.append(curr_sample_down)
    ups = np.concatenate(ups, axis=1)
    downs = np.concatenate(downs, axis=1)
    curr_sample = np.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return jnp.array(curr_sample)

