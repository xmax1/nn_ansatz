from nn_ansatz import *

# confirm antisymmetric

import jax
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad, pmap
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange

from functools import partial
import itertools

from nn_ansatz import *


cfg = setup(system='LiSolidBCC',
                    n_pre_it=0,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=64,
                    step_size=0.05,
                    n_ph=32,
                    scalar_inputs=False,
                    orbitals='anisotropic',
                    n_periodic_input=1,
                    opt='adam',
                    n_det=4,
                    print_every=50,
                    save_every=2500,
                    lr=1e-4,
                    n_it=30000,
                    debug=True, 
                    distribute=False)

logger = Logging(**cfg)

keys = rnd.PRNGKey(cfg['seed'])
if bool(os.environ.get('DISTRIBUTE')) is True:
    keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

mol = SystemAnsatz(**cfg)

vwf = create_wf(mol, orbitals=True)
params = initialise_params(mol, keys)

sampler = create_sampler(mol, vwf)

walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=None)

x = vwf(params, walkers)

def logabssumdet(orb_up: jnp.array, orb_down: jnp.array) -> jnp.array:
    s_up, log_up = jnp.linalg.slogdet(orb_up)
    s_down, log_down = jnp.linalg.slogdet(orb_down) if not orb_down is None else (jnp.ones_like(s_up), jnp.zeros_like(log_up))

    # logdet_sum = jnp.where(~jnp.isinf(log_up), log_up, -jnp.ones_like(log_up)*10.**10) + jnp.where(~jnp.isinf(log_down), log_down, -jnp.ones_like(log_down)*10.**10)
    logdet_sum = log_up + log_down
    logdet_max = jnp.max(logdet_sum)

    argument = s_up * s_down * jnp.exp(logdet_sum - logdet_max)
    sum_argument = jnp.sum(argument, axis=0)
    sign = jnp.sign(sum_argument)

    return jnp.log(jnp.abs(sum_argument)) + logdet_max, sign

print(x)