import os
# os.environ['JAX_PLATFORM_NAME']='cpu'
# os.environ['no_JIT'] = 'True'
# os.environ['XLA_FLAGS']="--xla_force_host_platform_device_count=4"
# os.environ['XLA_FLAGS']="--xla_dump_to=/tmp/foo"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import jax
print(jax.devices())

import numpy as np
from nn_ansatz import setup
from nn_ansatz import run_vmc

import jax.numpy as jnp

# orbitals = ['anisotropic',]
# scalar_inputs = [True, False]
# n_ps = [1,]

# for orbital in orbitals:
#     for scalar_input in scalar_inputs:
#         for n_p in n_ps:
#             if scalar_input:
#                 if n_p == 3:
#                     continue
#             name = 'Bowl_%s_si%s_np%i' % (orbital, str(scalar_input), n_p)
#             cfg = config = setup(system='LiSolidBCC',
#                         n_pre_it=0,
#                         n_walkers=512,
#                         n_layers=3,
#                         n_sh=64,
#                         step_size=0.02,
#                         n_ph=16,
#                         scalar_inputs=scalar_input,
#                         orbital_decay=orbital,
#                         n_periodic_input=n_p,
#                         opt='adam',
#                         n_det=4,
#                         print_every=100,
#                         save_every=2500,
#                         lr=1e-4,
#                         n_it=7500,
#                         name=name)
#             log = run_vmc(config)


step_sizes = [0.1, 0.05]
for step_size in step_sizes:
    name = 'BowlStep%.2f_%s_si%s_np%i' % (step_size, 'anisotropic', str(False), 1)
    cfg = config = setup(system='LiSolidBCC',
                n_pre_it=0,
                n_walkers=512,
                n_layers=3,
                n_sh=64,
                step_size=step_size,
                n_ph=16,
                scalar_inputs=False,
                orbital_decay='anisotropic',
                n_periodic_input=1,
                opt='adam',
                n_det=4,
                print_every=100,
                save_every=2500,
                lr=1e-4,
                n_it=7500,
                name=name)
    log = run_vmc(config)

# name = 'BowlStepLong%.2f_%s_si%s_np%i' % (0.02, 'anisotropic', str(False), 1)
# cfg = config = setup(system='LiSolidBCC',
#                 n_pre_it=0,
#                 n_walkers=512,
#                 n_layers=3,
#                 n_sh=64,
#                 step_size=0.02,
#                 n_ph=16,
#                 scalar_inputs=False,
#                 orbital_decay='anisotropic',
#                 n_periodic_input=1,
#                 opt='adam',
#                 n_det=4,
#                 print_every=100,
#                 save_every=2500,
#                 lr=1e-4,
#                 n_it=50000,
#                 name=name)
# log = run_vmc(config)