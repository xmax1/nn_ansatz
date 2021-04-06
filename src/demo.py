
from nn_ansatz import setup
from nn_ansatz import run_vmc

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
               damping=damping)

run_vmc(**config)

