
import sys
sys.path.append('..')
from nn_ansatz import setup, run_vmc, compare_einsum, approximate_energy

opts = ['adam', 'kfac']
n_it = 50000
for opt in opts:
    cfg = setup(system='HEG',
                n_pre_it=0,
                n_walkers=512,
                n_layers=2,
                n_sh=64,
                step_size=0.05,
                n_ph=32,
                scalar_inputs=False,
                orbitals='real_plane_waves',
                n_el = 7,
                n_periodic_input=1,
                opt=opt,
                n_det=1,
                density_parameter=1.,
                lr = 1e-4,
                print_every=1000,
                save_every=n_it,
                n_it=n_it,
                name='adam_vs_kfac')
    log = run_vmc(cfg)
    approximate_energy(cfg, load_it=n_it)


