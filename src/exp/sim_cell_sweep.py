

import sys
sys.path.append('..')
from nn_ansatz import setup, run_vmc, compare_einsum, approximate_energy

sim_cells = [(2, 2, 3)] # 
n_it = 25000

for sim_cell in sim_cells:
    cfg = setup(system='HSolid',
                        n_walkers=512,
                        n_layers=2,
                        n_sh=64,
                        n_ph=32,
                        orbitals='anisotropic',
                        simulation_cell=sim_cell,
                        input_activation_nonlinearity='bowl',
                        opt='kfac',
                        n_det=4,
                        n_it=n_it,
                        name='H_sim_cell_sweep')
    log = run_vmc(cfg)
    approximate_energy(cfg, load_it=n_it)
