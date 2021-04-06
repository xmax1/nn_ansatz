
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm import trange


from .sampling import create_sampler
from .ansatz import create_wf
from .parameters import initialise_params, initialise_d0s, expand_d0s
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
# from .utils import *
from .kfac import create_natural_gradients_fn, kfac
from .utils import Logging, load_pk, save_pk


def run_vmc(r_atoms=None,
            z_atoms=None,
            n_el=None,
            n_el_atoms=None,

            opt: str = 'kfac',
            lr: float = 1e-4,
            damping: float = 1e-4,
            norm_constraint: float = 1e-4,
            n_it: int = 1000,
            n_walkers: int = 1024,
            step_size: float = 0.02,

            n_layers: int = 2,
            n_sh: int = 64,
            n_ph: int = 16,
            n_det: int = 2,

            pre_lr: float = 1e-4,
            n_pre_it: int = 1000,
            load_pretrain: bool = False,
            pre_path: str = '',

            seed: int = 369,
            **kwargs):

    logger = Logging(**kwargs)

    key = rnd.PRNGKey(seed)

    mol = SystemAnsatz(r_atoms,
                       z_atoms,
                       n_el,
                       n_el_atoms=n_el_atoms,
                       n_layers=n_layers,
                       n_sh=n_sh,
                       n_ph=n_ph,
                       n_det=n_det,
                       step_size=step_size)

    wf, kfac_wf, wf_orbitals = create_wf(mol)
    params = initialise_params(key, mol)
    d0s = expand_d0s(initialise_d0s(mol), n_walkers)

    sampler, equilibrate = create_sampler(wf, mol, correlation_length=10)

    if load_pretrain:
        params, walkers = load_pk(pre_path)
    else:
        walkers = mol.initialise_walkers(n_walkers=n_walkers)
        params, walkers = pretrain_wf(params,
                                      wf,
                                      wf_orbitals,
                                      mol,
                                      walkers,
                                      n_it=n_pre_it,
                                      lr=pre_lr,
                                      n_eq_it=n_pre_it)
        save_pk([params, walkers], pre_path)

    params = initialise_params(key, mol)
    walkers = mol.initialise_walkers(n_walkers=n_walkers)

    grad_fn = create_grad_function(wf, mol)

    if opt == 'kfac':
        update, get_params, kfac_update, state = kfac(kfac_wf, wf, mol, params, walkers, d0s,
                                                       lr=lr,
                                                       damping=damping,
                                                       norm_constraint=norm_constraint)
    else:
        init, update, get_params = adam(lr)
        state = init(params)

    steps = trange(0, n_it, initial=0, total=n_it, desc='training', disable=None)
    for step in steps:
        key, subkey = rnd.split(key)

        walkers, acceptance, step_size = sampler(params, walkers, d0s, subkey, step_size)

        grads, e_locs = grad_fn(params, walkers, d0s)

        if opt == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers, d0s)

        # for g in grads:
        #     print(jnp.mean(jnp.abs(g)))

        # p1 = params
        state = update(step, grads, state)
        params = get_params(state)

        # t1, map1 = tree_util.tree_flatten(p1)
        # t2, map2 = tree_util.tree_flatten(params)
        #
        # for i, j in zip(t1, t2):
        #     print(jnp.mean(jnp.abs(i - j)))

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance)




