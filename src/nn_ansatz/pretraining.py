
import numpy as np
from tqdm.auto import trange
from functools import partial
import os 

from jax import value_and_grad, vmap, jit, pmap
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax.experimental.optimizers import adam

from .vmc import create_energy_fn
from .sampling import to_prob, create_sampler, equilibrate
from .parameters import initialise_d0s, expand_d0s, initialise_params
from .utils import save_pk, split_variables_for_pmap, key_gen
from .ansatz import create_wf, create_orbitals

def mia(arr):
    if type(arr) is float:
        return arr
    else:
        return jnp.mean(arr)


def pretrain_wf(mol,
                params,
                keys,
                sampler,
                walkers,
                pre_step_size: float = 0.02,
                n_pre_it: int = 1000,
                lr: float = 1e-4,
                n_eq_it: int = 500,
                pre_path=None,
                seed=1,
                **kwargs):
    pre_key = rnd.PRNGKey(seed)

    vwf = create_wf(mol)
    vwf_orbitals = create_wf(mol, orbitals=True)

    compute_local_energy = create_energy_fn(mol, vwf)
    if bool(os.environ.get('DISTRIBUTE')) is True:
        compute_local_energy = pmap(compute_local_energy, in_axes=(None, 0))

    loss_function, pre_sampler = create_pretrain_loss_and_sampler(mol, vwf, vwf_orbitals)
    loss_function = value_and_grad(loss_function)

    pre_walkers = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=True, n_it=n_eq_it)
    walkers = jnp.array(pre_walkers, copy=True)

    init, update, get_params = adam(lr)
    state = init(params)

    step_size = split_variables_for_pmap(walkers.shape[0], pre_step_size)
    steps = trange(0, n_pre_it, initial=0, total=n_pre_it, desc='pretraining', disable=None)


    for step in steps:
        pre_key, pre_subkey = rnd.split(pre_key)
        keys, subkey = key_gen(keys)

        loss_value, grads = loss_function(params, pre_walkers)
        state = update(step, grads, state)
        params = get_params(state)

        walkers, acceptance, step_size = sampler(params, walkers, subkey, step_size)
        e_locs = compute_local_energy(params, walkers)

        pre_walkers, mix_acceptance = pre_sampler(params, pre_walkers, pre_subkey, pre_step_size)
        e_locs_mixed = compute_local_energy(params, pre_walkers)

        print('step %i | e_mean %.2f | e_mixed %.2f | loss %.2f | wf_acc %.2f | mix_acc %.2f |'
              % (step, jnp.mean(e_locs), jnp.mean(e_locs_mixed), mia(loss_value), mia(acceptance), mia(mix_acceptance)))
        # steps.set_postfix(E=f'{e_locs.mean():.6f}')

    if not pre_path is None:
        save_pk([params, walkers], pre_path)

    return params, walkers


def create_pretrain_loss_and_sampler(mol, vwf, vwf_orbitals, correlation_length: int=10):

    if mol.pbc is True:
        if mol.system == 'HEG':
            real_plane_wave_orbitals, _ = create_orbitals(orbitals=mol.orbitals, n_el=mol.n_el, pbc=mol.pbc, basis=mol.basis, inv_basis=mol.inv_basis, einsum=mol.einsum)
            @jit
            def _hf_orbitals(walkers):
                orbs = real_plane_wave_orbitals(None, walkers, None)  # expects (n_k, n_el, n_el)
                return orbs, None
            _hf_orbitals = vmap(_hf_orbitals, in_axes=(0,))
            
    else:
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

        if not mol.system == 'HEG': walkers = _pyscf_call(walkers)
        up_dets, down_dets = _hf_orbitals(walkers)

        spin_ups = up_dets**2
        if not down_dets is None: spin_downs = down_dets**2

        p_up = jnp.diagonal(spin_ups, axis1=-2, axis2=-1).prod(-1)
        if not down_dets is None: p_down = jnp.diagonal(spin_downs, axis1=-2, axis2=-1).prod(-1)

        probabilities = p_up if mol.n_el == mol.n_up else p_up * p_down

        return probabilities

    def _loss_function(params, walkers):

        if len(walkers.shape) == 4: walkers = walkers.reshape((-1,) + walkers.shape[2:])

        wf_up_dets, wf_down_dets = vwf_orbitals(params, walkers)
        n_det = wf_up_dets.shape[1]

        if not mol.system == 'HEG': walkers = _pyscf_call(walkers)
        up_dets, down_dets = _hf_orbitals(walkers)
        
        up_dets = tile_labels(up_dets, n_det)
        if down_dets is not None: down_dets = tile_labels(down_dets, n_det)

        loss = mse_error(up_dets, wf_up_dets)
        if not down_dets is None: loss += mse_error(down_dets, wf_down_dets)
        return loss

    def _step_metropolis_hastings(params,
                                  curr_walkers_wf,
                                  curr_probs_wf,
                                  curr_walkers_hf,
                                  new_walkers_hf,
                                  curr_probs_hf,
                                  new_probs_hf,
                                  subkeys,
                                  step_size):
        shape = curr_walkers_wf.shape

        # next sample
        new_walkers_wf = curr_walkers_wf + rnd.normal(subkeys[0], shape) * step_size
        new_probs_wf = to_prob(vwf(params, new_walkers_wf))

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

    step_metropolis_hastings = _step_metropolis_hastings

    def _sample_metropolis_hastings(params, walkers, key, step_size):
        initial_shape = walkers.shape
        if len(walkers.shape) == 4: walkers = walkers.reshape((-1,) + walkers.shape[2:])

        curr_walkers_wf, curr_walkers_hf = jnp.split(walkers, 2, axis=0)

        shape = curr_walkers_wf.shape

        curr_probs_wf = to_prob(vwf(params, curr_walkers_wf))
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
                                         curr_walkers_hf,
                                         new_walkers_hf,
                                         curr_probs_hf,
                                         new_probs_hf,
                                         subkeys[1:],
                                         step_size)

            acceptance_total_wf += mask_wf.mean()
            acceptance_total_hf += mask_hf.mean()

        curr_walkers = jnp.concatenate([curr_walkers_wf, curr_walkers_hf], axis=0)
        walkers = rnd.permutation(key, curr_walkers).reshape(initial_shape)

        acc = (acceptance_total_wf + acceptance_total_hf) / (2 * float(correlation_length))
        return walkers, acceptance_total_hf / float(correlation_length)

    return _loss_function, _sample_metropolis_hastings


def mse_error(targets, outputs):
    return ((targets - outputs)**2).sum((1, 2, 3)).mean()


def tile_labels(label, n_k: int):
    x = jnp.repeat(label[:, None, ...], n_k, axis=1)
    return x


