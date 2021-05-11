
import jax
from jax import grad, lax, vmap, jit
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

# system
n_el = 4
r_atoms = jnp.array([[0.0, 0.0, 0.0]])
z_atoms = jnp.array([4.])

# training params
n_it = 10000
n_walkers = 32
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


params = initialise_params(key, mol)
d0s = expand_d0s(initialise_d0s(mol), n_walkers)
walkers = mol.initialise_walkers(n_walkers=n_walkers)


def local_kinetic_energy_i(wf):

    def _lapl_over_f(params, walkers, d0s):
        walkers = walkers.reshape(-1)
        n = walkers.shape[0]
        eye = jnp.eye(n, dtype=walkers.dtype)
        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y, d0s)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (walkers,), (eye[..., i],))
            return val + primal[i]**2 + tangent[i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


from nn_ansatz.ansatz import *


def create_wf(mol):
    n_up, n_down, r_atoms, n_el = mol.n_up, mol.n_down, mol.r_atoms, mol.n_el
    masks = create_masks(mol.n_atoms, mol.n_el, mol.n_up, mol.n_layers, mol.n_sh, mol.n_ph)

    def _wf_orbitals(params, walkers, d0s):

        if len(walkers.shape) == 1:  # this is a hack to get around the jvp
            walkers = walkers.reshape(n_up + n_down, 3)

        activations = []

        ae_vectors = compute_ae_vectors_i(walkers, r_atoms)

        single, pairwise = compute_inputs_i(walkers, ae_vectors)

        single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *masks[0])

        split = linear_split(params['split0'], split, activations, d0s['split0'])
        single = linear(params['s0'], single_mixed, split, activations, d0s['s0'])
        pairwise = linear_pairwise(params['p0'], pairwise, activations, d0s['p0'])

        for (split_params, s_params, p_params), (split_per, s_per, p_per), mask \
                in zip(params['intermediate'], d0s['intermediate'], masks[1:]):
            single_mixed, split = mixer_i(single, pairwise, n_el, n_up, n_down, *mask)

            split = linear_split(split_params, split, activations, split_per)
            single = linear(s_params, single_mixed, split, activations, s_per) + single
            pairwise = linear_pairwise(p_params, pairwise, activations, p_per) + pairwise

        ae_up, ae_down = jnp.split(ae_vectors, [n_up], axis=0)
        data_up, data_down = jnp.split(single, [n_up], axis=0)

        factor_up = env_linear_i(params['envelopes']['linear'][0], data_up, activations, d0s['envelopes']['linear'][0])
        factor_down = env_linear_i(params['envelopes']['linear'][1], data_down, activations, d0s['envelopes']['linear'][1])

        exp_up = env_sigma_i(params['envelopes']['sigma'][0], ae_up, activations, d0s['envelopes']['sigma'][0])
        exp_down = env_sigma_i(params['envelopes']['sigma'][1], ae_down, activations, d0s['envelopes']['sigma'][1])

        orb_up = env_pi_i(params['envelopes']['pi'][0], factor_up, exp_up, activations, d0s['envelopes']['pi'][0])
        orb_down = env_pi_i(params['envelopes']['pi'][1], factor_down, exp_down, activations, d0s['envelopes']['pi'][0])
        return orb_up, orb_down, activations

    def _kfac_wf(params, walkers, d0s):

        orb_up, orb_down, activations = _wf_orbitals(params, walkers, d0s)
        log_psi = logabssumdet(orb_up, orb_down)
        return log_psi, activations

    def _wf(params, walkers, d0s):
        orb_up, orb_down, activations = _wf_orbitals(params, walkers, d0s)
        log_psi = logabssumdet(orb_up, orb_down)
        return log_psi

    wf_orbitals = remove_aux(_wf_orbitals, axis=1)

    return _wf, _kfac_wf, wf_orbitals


def local_kinetic_energy_i(wf):

    def _lapl_over_f(params, walkers, d0s):
        walkers = walkers.reshape(-1)
        n = walkers.shape[0]
        eye = jnp.eye(n, dtype=walkers.dtype)
        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y, d0s)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (walkers,), (eye[..., i],))
            return val + primal[i]**2 + tangent[i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f



"""
the energy function can't cope with 2 outputs

removing the aux doesn't work

this is the current workaround

"""





















exit()

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











