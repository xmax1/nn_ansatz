import os
os.environ['JAX_PLATFORM_NAME']='cpu'
os.environ['no_JIT'] = 'True'
# os.environ['XLA_FLAGS']="--xla_force_host_platform_device_count=4"
# os.environ['XLA_FLAGS']="--xla_dump_to=/tmp/foo"

import jax
print(jax.devices())

import numpy as np
from nn_ansatz import setup
from nn_ansatz import run_vmc

import jax.numpy as jnp

lr, damping, nc = 1e-4, 1e-4, 1e-4

config = setup(system='LiSolid',
               n_pre_it=501,
               n_walkers=128,
               n_layers=2,
               n_sh=32,
               n_ph=8,
               opt='kfac',
               n_det=4,
               print_every=1,
               save_every=5000,
               lr=1e-3,
               n_it=1000,
               norm_constraint=1e-4,
               damping=1e-3,
               exp=True,
               name='bulk_minim_interactionsandpotential',
               kappa = 0.5,
               real_cut = 5,
               reciprocal_cut = 5)

run_vmc(**config)

