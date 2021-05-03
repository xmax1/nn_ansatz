import numpy as np
from nn_ansatz import setup
from nn_ansatz import run_vmc

import jax.numpy as jnp

lr, damping, nc = 1e-4, 1e-4, 1e-4

config = setup(system='LiSolid',
               #pretrain=True,
               n_pre_it=501,
               n_walkers=1024,
               n_layers=2,
               n_sh=64,
               n_ph=16,
               opt='adam',
               n_det=8,
               print_every=1,
               save_every=5000,
               lr=1e-3,
               n_it=1000,
               norm_constraint=1e-4,
               damping=1e-3,
               exp=True,
               name='sampler_fix')

run_vmc(**config)

