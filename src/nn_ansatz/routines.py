
import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange


from .sampling import create_sampler
from .ansatz import create_wf
from .parameters import initialise_params, initialise_d0s, expand_d0s
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
# from .utils import *
from .optimisers import create_natural_gradients_fn, kfac
from .utils import Logging, load_pk, save_pk


def run_vmc(r_atoms=None,
            z_atoms=None,
            n_el=None,
            n_el_atoms=None,
            periodic_boundaries=False,
            cell_basis=None,
            unit_cell_length=None,

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
            pretrain: bool = False,
            pre_path: str = '',

            seed: int = 369,
            **kwargs):

    logger = Logging(**kwargs)

    key = rnd.PRNGKey(seed)

    mol = SystemAnsatz(r_atoms,
                       z_atoms,
                       n_el,
                       unit_cell_length=unit_cell_length,
                       cell_basis=cell_basis,
                       periodic_boundaries=periodic_boundaries,
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

    walkers = mol.initialise_walkers(n_walkers=n_walkers)

    if pretrain:  # this logic is ugly but need to include case where we want to skip pretraining
        if load_pretrain:
            params, walkers = load_pk(pre_path)
            walkers = mol.initialise_walkers(walkers=walkers,
                                             n_walkers=n_walkers,
                                             equilibrate=equilibrate,
                                             params=params,
                                             d0s=d0s)
        else:
            params, walkers = pretrain_wf(params,
                                          wf,
                                          wf_orbitals,
                                          mol,
                                          walkers,
                                          n_it=n_pre_it,
                                          lr=pre_lr,
                                          n_eq_it=n_pre_it,
                                          pre_path=pre_path)

    walkers = mol.initialise_walkers(walkers=walkers,
                                     n_walkers=n_walkers,
                                     equilibrate=equilibrate,
                                     params=params,
                                     d0s=d0s)

    grad_fn = create_grad_function(wf, mol)

    if opt == 'kfac':
        update, get_params, kfac_update, state = kfac(kfac_wf, wf, mol, params, walkers, d0s,
                                                      lr=lr,
                                                      damping=damping,
                                                      norm_constraint=norm_constraint)
    elif opt == 'adam':
        init, update, get_params = adam(lr)
        update = jit(update)
        state = init(params)
    else:
        exit('Optimiser not available')

    steps = trange(0, n_it, initial=0, total=n_it, desc='training', disable=None)
    for step in steps:
        key, subkey = rnd.split(key)

        walkers, acceptance, step_size = sampler(params, walkers, d0s, subkey, step_size)

        grads, e_locs = grad_fn(params, walkers, d0s)

        if opt == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers, d0s)

        state = update(step, grads, state)
        params = get_params(state)

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance)


