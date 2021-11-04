
from nn_ansatz import setup, run_vmc, compare_einsum

cfg = setup(system='LiSolidBCC',
                    n_pre_it=0,
                    n_walkers=512,
                    n_layers=3,
                    n_sh=64,
                    step_size=0.05,
                    n_ph=32,
                    scalar_inputs=False,
                    orbitals='anisotropic',
                    n_periodic_input=1,
                    opt='adam',
                    n_det = 4,
                    lr = 1e-4,
                    print_every=50,
                    save_every=2500,
                    n_it=30000,
                    einsum=True,
                    name='ein_compare_ein')
log = run_vmc(cfg)

# cfg = setup(system='LiSolidBCC',
#                 n_pre_it=0,
#                 n_walkeroldtransform
#                 n_layers=2,
#                 n_sh=64,
#                 step_size=0.05,
#                 n_ph=32,
#                 scalar_inputs=False,
#                 orbitals='anisotropic',
#                 einsum=False,
#                 n_periodic_input=1,
#                 opt='adam',
#                 n_det=4,
#                 print_every=50,
#                 save_every=2500,
#                 lr=1e-4,
#                 n_it=30000)

# compare_einsum(cfg)
