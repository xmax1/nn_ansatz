
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
from .utils import Logging, load_pk, save_pk, key_gen


def split_variables_for_pmap(n_devices, *args):
    for i in range(len(args))[:-1]:
        assert len(args[i]) == len(args[i + 1])

    assert len(args[0]) % n_devices == 0

    new_args = []
    for arg in args:
        shape = arg.shape
        new_args.append(arg.reshape(n_devices, shape[0] // n_devices, *shape[1:]))

    if len(args) == 1:
        return new_args[0]
    return new_args


def run_vmc(opt: str = 'kfac',
            lr: float = 1e-4,
            damping: float = 1e-4,
            norm_constraint: float = 1e-4,
            n_it: int = 1000,
            n_walkers: int = 1024,
            step_size=None,

            pre_lr: float = 1e-4,
            n_pre_it: int = 1000,
            load_pretrain: bool = False,
            pretrain: bool = False,
            pre_path: str = '',

            seed: int = 369,
            **kwargs):

    logger = Logging(**kwargs)

    key = rnd.PRNGKey(seed)

    mol = SystemAnsatz(**kwargs)

    wf, kfac_wf, wf_orbitals = create_wf(mol)
    params = initialise_params(key, mol)
    d0s = initialise_d0s(mol)

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
    walkers = split_variables_for_pmap(kwargs['n_devices'], walkers)

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
    keys = rnd.split(key, kwargs['n_devices'])

    for step in steps:
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, d0s, subkeys, step_size)
        step_size = step_size[0]

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
                   acceptance=acceptance[0])


