
from nn_ansatz import setup, run_vmc

cfg = setup(system='LiSolidBCC',
            n_pre_it=0,
            n_walkers=256,
            n_layers=2,
            n_sh=32,
            step_size=0.02,
            n_ph=8,
            scalar_inputs=False,
            orbitals='anisotropic',
            n_periodic_input=1,
            opt='kfac',
            n_det=7,
            print_every=100,
            save_every=2500,
            lr=1e-4,
            n_it=100,
            name='gputest')
log = run_vmc(cfg)

