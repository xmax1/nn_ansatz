import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad, pmap
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange

from functools import partial

import os

from nn_ansatz import *

from fabric import Connection
import subprocess as sub
import sys
sys.path.append('/home/energy/amawi/projects/nn_ansatz/src')
from nn_ansatz.vmc import create_potential_energy

mol = SystemAnsatz(system=None,
                 r_atoms=None,
                 z_atoms=None,
                 n_el=7,
                 basis=jnp.eye(3),
                 inv_basis=None,
                 pbc=True,
                 spin_polarized=True,
                 density_parameter=1.,
                 real_cut=None,
                 reciprocal_cut=None,
                 kappa=0.75,
                 simulation_cell = (1, 1, 1),
                 scalar_inputs=False,
                 orbitals='real_plane_waves',
                 device='gpu',
                 dtype=jnp.float32,
                 scale_cell=1.,
                 print_ansatz=True,
                 n_walkers_per_device=1024,
                 n_devices=1)

pe_fn = create_potential_energy(mol, find_kappa=False)

walkers = jnp.array([[2.1526934277668843,         2.4560512832645855,          0.51311075081762780],    
                    [1.3791267107711942    ,     1.6234439638535862    ,      3.0667537383069945  ,]  
                    [1.4656891738654672    ,     0.43665165495674785   ,     1.4862060883914250   ,] 
                    [0.53729657298062961   ,     1.4812046409522104    ,     0.78584559911787721  ,]  
                    [1.1167108011660005    ,     2.6942379435411468    ,      1.2266702700217422  ,]  
                    [1.2572663823111512    ,     0.25995966342111387   ,      0.24667337122245669 ,]   
                    [1.9453890650886052    ,     1.1690075379781248    ,       2.3061978534074310,]])
pe = pe_fn(walkers, mol.r_atoms, mol.z_atoms)

print(pe)

# x = sub.Popen("screen -dmS test bash -c 'CUDA_VISIBLE_DEVICES=\'3\' python /home/amawi/projects/nn_ansatz/src/run_with_args.py'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# print(x)
# import time

# time.sleep(60)
# x = sub.run('echo $LD_LIBRARY_PATH', shell=True)
# launch_path = '/home/amawi/projects/nn_ansatz/src/exp/launch_exp.sh'
# path = '/home/amawi/projects/nn_ansatz/src/run_with_args.py'

# host = 'titan02'
# user = 'amawi'
# cmd = 'CUDA_VISIBLE_DEVICES="3" python ' + '/home/amawi/projects/nn_ansatz/src/run_with_args.py'
# out = subprocess.Popen("ssh {user}@{host} {cmd} >null 2>&1".format(user=user, host=host, cmd=cmd), shell=True)
# x = sub.run('screen -dmS test %s' % cmd, shell=True)

# x = Connection('amawi@titan02')
# x.run('sh %s' % launch_path)
# x.run('source ~/.bashrc')
# x.run('module load CUDA/11.2')
# x.run('echo $LD_LIBRARY_PATH')
# x.run('echo max')

# cfg = setup(system='HEG',
#             n_walkers=512,
#             n_layers=2,
#             n_sh=64,
#             step_size=0.05,
#             n_ph=32,
#             orbitals='real_plane_waves',
#             n_el = 7,
#             input_activation_nonlinearity='sin+cos+bowl',
#             n_periodic_input=1,
#             opt='kfac',
#             n_det=1,
#             density_parameter=1.,
#             lr = 1e-4,
#             n_it=1000,
#             name='111121/pre_test')


# mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, None)
# params, walkers = pretrain_wf(mol, params, keys, sampler, walkers, n_eq_it=100)





### ENERGY DEBUG
# cfg = load_pk('/home/amawi/projects/nn_ansatz/src/exp/experiments/HEG/101121/heg_test/kfac_1lr-4_1d-3_1nc-4_m512_s64_p32_l2_det4/run1/config1.pk')
# # os.environ['DISTRIBUTE'] =  'no'
# walkers = None

# mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)
# walkers = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=True, n_it=200)
# energy_function = create_energy_fn(mol, vwf, separate=True)

# if bool(os.environ.get('DISTRIBUTE')) is True:
#     energy_function = pmap(energy_function, in_axes=(None, 0))

# local_kinetic_energy = create_local_kinetic_energy(vwf)
# if bool(os.environ.get('DISTRIBUTE')) is True:
#     local_kinetic_energy = pmap(local_kinetic_energy, in_axes=(None, 0))

# ke = local_kinetic_energy(params, walkers)