from ase import Atoms
from ase.visualize import view
from gpaw import GPAW, PW

name = 'He-pc-4'
a = 4  # fcc lattice parameter
b = a / 2
bulk = Atoms('He',  positions=[(b, b, b)],
             cell=[[a, 0, 0],
                   [0, a, 0],
                   [0, 0, a]],
             pbc=True)
k = 1
calc = GPAW(mode=PW(700),       # cutoff
            kpts=(k, k, k),     # k-points
            txt=name + '.txt')  # output file
bulk.calc = calc
energy = bulk.get_potential_energy()
print(energy)
# calc.write(name + '.gpw')
# print('Energy:', energy, 'eV')