
from nn_ansatx import setup
from nn_ansatx import run_vmc

lr, damping, nc = 1e-4, 1e-4, 1e-4

config = setup(system='Be',
               n_pre_it=500,
               n_walkers=512,
               n_layers=2,
               n_sh=64,
               n_ph=16,
               opt='kfac',
               n_det=8,
               print_every=1,
               save_every=5000,
               lr=lr,
               n_it=1000,
               norm_constraint=nc,
               damping=damping,
               name='kfac_debug',
               exp=True)

run_vmc(**config)

# for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
#     for damping in [1e-1, 1e-2, 1e-3, 1e-4]:
#         for nc in [1e-1, 1e-2, 1e-3, 1e-4]:
#             config = setup(system='Be',
#                            n_el=4,
#                            n_pre_it=500,
#                            n_walkers=1024,
#                            opt='kfac',
#                            print_every=1,
#                            save_every=5000,
#                            lr=lr,
#                            n_it=1000,
#                            norm_constraint=nc,
#                            damping=damping,
#                            name='lr_damping_sweep',
#                            exp=True)
#
#             run_vmc(**config)

