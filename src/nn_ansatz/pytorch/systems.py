import torch as tc
import math
from pyscf import dft, gto
from pyscf.scf import RHF


def create_atom(r_atoms, z_atom):
    r_atoms = r_atoms.mean(dim=0)
    atom = []
    for charge, coord in zip(z_atom, r_atoms):
        atom.append([int(charge.detach().cpu().numpy()), tuple(coord.detach().cpu().numpy())])
    return atom



class Molecule():
    def __init__(self,
                 r_atoms,
                 z_atoms,
                 n_el,
                 n_layers=2,
                 n_sh=64,
                 n_ph=16,
                 n_det=2,
                 n_el_atoms=None,
                 n_up=None,
                 basis='sto3g',
                 device='cpu',
                 dtype=tc.float32):
        self.device, self.dtype = device, dtype

        r_atoms = r_atoms.to(device=device, dtype=dtype)
        z_atoms = z_atoms.to(device=device, dtype=dtype)

        if n_up is None:
            n_up = math.ceil(n_el / 2.)
        n_down = n_el - n_up

        self._n_el = n_el
        self._n_up = n_up
        self._n_down = n_down
        self._spin = n_up - n_down  # will this be different for molecules?

        self._n_atoms = r_atoms.shape[1]
        self._atom = create_atom(r_atoms, z_atoms)
        self._charge = int(tc.sum(z_atoms)) - n_el
        self._z_atoms = z_atoms
        self._r_atoms = r_atoms

        if n_el_atoms is None:
            self._n_el_atoms = [int(x) for x in z_atoms.detach().cpu().numpy()]

        print('System: \n',
              'Device  = %s \n' % device,
              'dtype   = %s \n' % str(dtype),
              'n_atoms = %i \n' % self.n_atoms,
              'n_up    = %i \n' % n_up,
              'n_down  = %i \n' % n_down)

        mol = gto.M(
            atom=self._atom,
            unit='bohr',
            basis=basis,
            charge=self._charge,
            spin=self._spin,
            cart=True,
        )
        mol.build()

        mf = RHF(mol)
        mf.kernel()

        self.mf = mf
        self.pyscf_mol = mol
        self.moT = tc.from_numpy(mf.mo_coeff.T).to(device=device, dtype=dtype)

        self.n_layers = n_layers
        self.n_sh = n_sh
        self.n_ph = n_ph
        self.n_det = n_det

    @property
    def atom(self):
        return self._atom

    @property
    def n_el(self):
        return self._n_el

    @property
    def n_el_atoms(self):
        return self._n_el_atoms

    @property
    def n_up(self):
        return self._n_up

    @property
    def n_down(self):
        return self._n_down

    @property
    def charge(self):
        return self._charge

    @property
    def spin(self):
        return self._spin

    @property
    def n_atoms(self):
        return self._n_atoms

    @property
    def r_atoms(self):
        return self._r_atoms

    @property
    def atom_positions(self):
        return [x.detach().cpu().numpy() for x in self._r_atoms.sum(0).split(self.n_atoms, dim=0)]

    @property
    def z_atoms(self):
        return self._z_atoms

    def initialise_walkers(self, n_walkers=1024):
        return tc.normal(0., 1., (n_walkers, self.n_el, 3), device=self.device, dtype=self.dtype)


if __name__ == '__main__':
    import numpy as np
    import torch as tc
    from pyscf import scf

    n_el, n_up, n_down = 4, 2, 2

    r_atoms = tc.from_numpy(np.zeros((1, 1, 3)))
    z_atoms = tc.from_numpy(np.ones((1,)) * n_el)
    mol = Molecule(r_atoms, z_atoms, n_el)

    atom = create_atom(r_atoms, z_atoms)
    mol = gto.M(
        atom=atom,
        unit='bohr',
        basis='sto3g',
        charge=int(tc.sum(z_atoms)) - n_el,
        spin=n_up-n_down,
        cart=True,
    )
    # mol.build()  # needs to be called if something changes about the molecule

    mf = RHF(mol)
    mf.kernel()
    print('here')
    mo = mf['mo_coeff']

    # this function is from the original work
    # the mo returned is what filip termed the 'super twist matrix'
    # they are the molecular orbital coefficients
    # conv, e, mo_e, mo, mo_occ = scf.rhf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))

    print('end')

