
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam

from tqdm import trange

from nn_ansatx import create_sampler
from nn_ansatx import create_wf
from nn_ansatx import initialise_params
from nn_ansatx import SystemAnsatz
from nn_ansatx import pretrain
from nn_ansatx import create_grad_function
from nn_ansatx import create_energy_function

# randomness
key = rnd.PRNGKey(1)
key, *subkeys = rnd.split(key, num=3)

# system
n_el = 4
r_atoms = jnp.array([[0.0, 0.0, 0.0]])
z_atoms = jnp.array([4.])

# training params
n_it = 10000
n_pre_it = 10
lr = 1e-4
pre_lr = 1e-4

step_size = 0.02
mol = SystemAnsatz(r_atoms,
                   z_atoms,
                   n_el,
                   n_layers=2,
                   n_sh=64,
                   n_ph=16,
                   n_det=2,
                   step_size=step_size)

wf, wf_orbitals = create_wf(mol)
params = initialise_params(subkeys[0], mol)

sampler, equilibrate = create_sampler(wf, mol, correlation_length=10)

walkers = mol.initialise_walkers(n_walkers=2048)

params, walkers = pretrain(params,
                           wf,
                           wf_orbitals,
                           mol,
                           walkers,
                           n_it=n_pre_it,
                           lr=pre_lr)


# training
grad_fn = create_grad_function(wf, mol)
init, update, get_params = adam(lr)
state = init(params)

steps = trange(0, n_it, initial=0, total=n_it, desc='training', disable=None)
for step in steps:
    key, subkey = rnd.split(key)

    walkers, acc, step_size = sampler(params, walkers, subkey, step_size)

    grads, e_locs = grad_fn(params, walkers)

    state = update(step, grads, state)
    params = get_params(state)

    #steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
    print('step %i | energy %.4f | acceptance %.2f' % (step, jnp.mean(e_locs), acc))




