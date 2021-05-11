import numpy as np
from nn_ansatz import setup
from nn_ansatz import run_vmc

import jax.numpy as jnp

lr, damping, nc = 1e-4, 1e-4, 1e-4

config = setup(system='LiSolid',
               #pretrain=True,
               n_pre_it=501,
               n_walkers=128,
               n_layers=2,
               n_sh=32,
               n_ph=8,
               opt='adam',
               n_det=4,
               print_every=1,
               save_every=5000,
               lr=1e-3,
               n_it=10000,
               norm_constraint=1e-4,
               damping=1e-3,
               exp=True,
               name='mic_w_interaction',
               kappa = 1.,
               real_cut = 6.,
               reciprocal_cut = 13)

run_vmc(**config)

