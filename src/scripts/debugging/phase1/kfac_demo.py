
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_flatten, tree_unflatten

from tqdm import trange

from nn_ansatx import create_sampler
from nn_ansatx import create_wf
from nn_ansatx import initialise_params
from nn_ansatx import initialise_d0s, expand_d0s
from nn_ansatx import SystemAnsatz
from nn_ansatx import pretrain
from nn_ansatx import create_grad_function
from nn_ansatx import remove_aux
from nn_ansatx import create_energy_fn
from nn_ansatx import create_natural_gradients_fn

# randomness
key = rnd.PRNGKey(1)
print(key)

# system
n_el = 4
r_atoms = jnp.array([[0.0, 0.0, 0.0]])
z_atoms = jnp.array([4.])

# training params
n_it = 10000
n_walkers = 1024
n_pre_it = 1
lr = 1e-4
pre_lr = 1e-4
damping = 1e-3
norm_constraint = 1e-4

step_size = 0.02
mol = SystemAnsatz(r_atoms,
                   z_atoms,
                   n_el,
                   n_layers=2,
                   n_sh=64,
                   n_ph=16,
                   n_det=2,
                   step_size=step_size)

wf, kfac_wf, wf_orbitals = create_wf(mol)
params = initialise_params(key, mol)
d0s = expand_d0s(initialise_d0s(mol), n_walkers)

sampler, equilibrate = create_sampler(wf, mol, correlation_length=10)

walkers = mol.initialise_walkers(n_walkers=n_walkers)

params, walkers = pretrain(params,
                           wf,
                           wf_orbitals,
                           mol,
                           walkers,
                           n_it=n_pre_it,
                           lr=pre_lr,
                           n_eq_it=n_pre_it)

compute_local_energy = create_energy_fn(wf, mol)


kfac, maas, msss = create_natural_gradients_fn(kfac_wf, wf, mol, params, walkers, d0s)





steps = trange(0, n_it, initial=0, total=n_it, desc='training', disable=None)
for step in steps:
    key, subkey = rnd.split(key)

    walkers, acceptance, step_size = sampler(params, walkers, d0s, subkey, step_size)
    e_locs = compute_local_energy(params, walkers, d0s)
    ngs, maas, msss = kfac(step, params, walkers, d0s, maas, msss, lr, damping, norm_constraint)

    params = update(params, ngs)

    #steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
    print('step %i | energy %.4f | acceptance %.2f' % (step, jnp.mean(e_locs), acceptance))









