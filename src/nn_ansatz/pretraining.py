
import numpy as np
from tqdm.auto import trange
import pickle as pk
import time

from jax import value_and_grad, vmap, jit
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax.experimental.optimizers import adam

from .vmc import create_energy_fn
from .sampling import to_prob, create_sampler
from .parameters import initialise_d0s, expand_d0s
from .utils import save_pk

def pretrain_wf(params,
                wf,
                wf_orbitals,
                mol,
                walkers,
                step_size: float = 0.02,
                n_it: int = 1000,
                lr: float = 1e-4,
                n_eq_it: int = 500,
                pre_path=None,
                seed=1,
                **kwargs):
    key = rnd.PRNGKey(seed)

    print('READ ME: Be careful here, '
          '(see env_sigma_i() in ferminet) wf samples and mixed samples energies diverge'
          'if the sigma and pi parameters are not set up in a very particular way\n')

    time.sleep(1)
    d0s_pre = expand_d0s(initialise_d0s(mol), len(walkers)//2)
    d0s = expand_d0s(initialise_d0s(mol), len(walkers))

    compute_local_energy = create_energy_fn(wf, mol)
    loss_function, sampler = create_pretrain_loss_and_sampler(wf, wf_orbitals, mol)
    loss_function = value_and_grad(loss_function)
    wf_sampler, equilibrate = create_sampler(wf, mol, correlation_length=10)

    walkers = equilibrate(params, walkers, d0s, key, n_it=n_eq_it, step_size=step_size)
    wf_walkers = jnp.array(walkers, copy=True)

    init, update, get_params = adam(lr)
    state = init(params)

    steps = trange(0, n_it, initial=0, total=n_it, desc='pretraining', disable=None)
    for step in steps:
        key, *subkeys = rnd.split(key, num=3)

        loss_value, grads = loss_function(params, walkers, d0s)
        state = update(step, grads, state)
        params = get_params(state)

        wf_walkers, acceptance, step_size = wf_sampler(params, wf_walkers, d0s, subkeys[0], step_size)
        e_locs = compute_local_energy(params, wf_walkers, d0s)

        walkers, mix_acceptance = sampler(params, walkers, d0s_pre, subkeys[1], step_size)
        e_locs_mixed = compute_local_energy(params, walkers, d0s)

        print('step %i | e_mean %.2f | e_mixed %.2f | loss %.2f | wf_acc %.2f | mix_acc %.2f |'
              % (step, jnp.mean(e_locs), jnp.mean(e_locs_mixed), loss_value, acceptance, mix_acceptance))
        # steps.set_postfix(E=f'{e_locs.mean():.6f}')

    if not pre_path is None:
        save_pk([params, walkers], pre_path)

    return params, wf_walkers


def create_pretrain_loss_and_sampler(wf, wf_orbitals, mol, correlation_length: int=10):

    vwf = jit(vmap(wf, in_axes=(None, 0, 0)))
    vwf_orbitals = jit(vmap(wf_orbitals, in_axes=(None, 0, 0)))

    def _pyscf_call(walkers):
        walkers = walkers.reshape((-1, 3))
        device = walkers.device()
        walkers = np.array(walkers)
        ao_values = mol.pyscf_mol.eval_gto("GTOval_cart", walkers)
        ao_values = jax.device_put(jnp.array(ao_values), device)
        return ao_values.reshape((-1, mol.n_el, ao_values.shape[-1]))

    def _hf_orbitals(ao_values):

        spin_up = jnp.stack([(mol.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
             for orb_number in range(mol.n_up) for el_number in
             range(mol.n_up)], axis=1).reshape((-1, mol.n_up, mol.n_up))

        spin_down = jnp.stack([(mol.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
                            for orb_number in range(mol.n_down) for el_number in
                            range(mol.n_up, mol.n_el)], axis=1).reshape((-1, mol.n_down, mol.n_down))

        return spin_up, spin_down

    _hf_orbitals = jit(_hf_orbitals)

    def _compute_orbital_probability(walkers):

        ao_values = _pyscf_call(walkers)
        up_dets, down_dets = _hf_orbitals(ao_values)

        spin_ups = up_dets**2
        spin_downs = down_dets**2

        p_up = jnp.diagonal(spin_ups, axis1=-2, axis2=-1).prod(-1)
        p_down = jnp.diagonal(spin_downs, axis1=-2, axis2=-1).prod(-1)

        probabilities = p_up * p_down

        return probabilities

    def _loss_function(params, walkers, d0s):

        wf_up_dets, wf_down_dets = vwf_orbitals(params, walkers, d0s)
        n_det = wf_up_dets.shape[1]

        ao_values = _pyscf_call(walkers)
        up_dets, down_dets = _hf_orbitals(ao_values)
        up_dets = tile_labels(up_dets, n_det)
        down_dets = tile_labels(down_dets, n_det)

        loss = mse_error(up_dets, wf_up_dets)
        loss += mse_error(down_dets, wf_down_dets)
        return loss

    def _step_metropolis_hastings(params,
                                  curr_walkers_wf,
                                  curr_probs_wf,
                                  d0s,
                                  curr_walkers_hf,
                                  new_walkers_hf,
                                  curr_probs_hf,
                                  new_probs_hf,
                                  subkeys,
                                  step_size):
        shape = curr_walkers_wf.shape

        # next sample
        new_walkers_wf = curr_walkers_wf + rnd.normal(subkeys[0], shape) * step_size
        new_probs_wf = to_prob(vwf(params, new_walkers_wf, d0s))

        # update sample
        alpha_wf = new_probs_wf / curr_probs_wf
        alpha_hf = new_probs_hf / curr_probs_hf

        # masks
        mask_wf = alpha_wf > rnd.uniform(subkeys[1], (shape[0],))
        mask_hf = alpha_hf > rnd.uniform(subkeys[2], (shape[0],))

        # update walkers & probs
        curr_walkers_wf = jnp.where(mask_wf[:, None, None], new_walkers_wf, curr_walkers_wf)
        curr_walkers_hf = jnp.where(mask_hf[:, None, None], new_walkers_hf, curr_walkers_hf)

        curr_probs_wf = jnp.where(mask_wf, new_probs_wf, curr_probs_wf)
        curr_probs_hf = jnp.where(mask_hf, new_probs_hf, curr_probs_hf)

        return curr_walkers_wf, curr_probs_wf, curr_walkers_hf, curr_probs_hf, mask_wf, mask_hf

    step_metropolis_hastings = jit(_step_metropolis_hastings)

    def _sample_metropolis_hastings(params, walkers, d0s, key, step_size):

        curr_walkers_wf, curr_walkers_hf = jnp.split(walkers, 2, axis=0)

        shape = curr_walkers_wf.shape

        curr_probs_wf = to_prob(vwf(params, curr_walkers_wf, d0s))
        curr_probs_hf = _compute_orbital_probability(curr_walkers_hf)

        acceptance_total_wf = 0.
        acceptance_total_hf = 0.
        for _ in range(correlation_length):
            key, *subkeys = rnd.split(key, num=5)

            new_walkers_hf = curr_walkers_hf + rnd.normal(subkeys[0], shape) * step_size
            new_probs_hf = _compute_orbital_probability(new_walkers_hf)  # can't be jit

            curr_walkers_wf, curr_probs_wf, curr_walkers_hf, curr_probs_hf, mask_wf, mask_hf = \
                step_metropolis_hastings(params,
                                         curr_walkers_wf,
                                         curr_probs_wf,
                                         d0s,
                                         curr_walkers_hf,
                                         new_walkers_hf,
                                         curr_probs_hf,
                                         new_probs_hf,
                                         subkeys[1:],
                                         step_size)

            acceptance_total_wf += mask_wf.mean()
            acceptance_total_hf += mask_hf.mean()

        walkers = jnp.concatenate([curr_walkers_wf, curr_walkers_hf], axis=0)
        walkers = rnd.permutation(key, walkers)

        acc = (acceptance_total_wf + acceptance_total_hf) / (2 * float(correlation_length))
        return walkers, acceptance_total_hf / float(correlation_length)

    return _loss_function, _sample_metropolis_hastings


def mse_error(targets, outputs):
    return ((targets - outputs)**2).sum((1, 2, 3)).mean()


def tile_labels(label, n_k: int):
    x = jnp.repeat(label[:, None, ...], n_k, axis=1)
    return x


