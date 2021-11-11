
import sys
sys.path.append('..')
from nn_ansatz import setup, run_vmc, compare_einsum, approximate_energy

densities = [3., 4., 5.] # 3., 4., 5.
n_it = 25000

for density in densities:
    cfg = setup(system='HEG',
                n_walkers=512,
                n_layers=2,
                n_sh=64,
                n_ph=32,
                orbitals='real_plane_waves',
                input_activation_nonlinearity='bowl',
                n_el = 7,
                opt='kfac',
                n_det=1,
                density_parameter=density,
                n_it=n_it,
                name='bowl_density_sweep_HEG')
    log = run_vmc(cfg)
    approximate_energy(cfg, load_it=n_it)


