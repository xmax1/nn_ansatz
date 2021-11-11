
import sys
sys.path.append('..')
from nn_ansatz import setup, run_vmc, compare_einsum, approximate_energy

n_it = 50000
nonlinearities = ['tanh', 'sin']
for nonlinearity in nonlinearities:
    cfg = setup(system='LiSolidBCC',
                n_pre_it=0,
                n_walkers=512,
                n_layers=2,
                n_sh=64,
                step_size=0.05,
                n_ph=32,
                scalar_inputs=False,
                orbitals='anisotropic',
                n_periodic_input=1,
                opt='kfac',
                n_det=4,
                nonlinearity=nonlinearity,
                density_parameter=1.,
                lr = 1e-4,
                print_every=1000,
                save_every=n_it,
                n_it=n_it,
                name='nonlinearity_compare_Li')
    log = run_vmc(cfg)
    approximate_energy(cfg, load_it=n_it)


