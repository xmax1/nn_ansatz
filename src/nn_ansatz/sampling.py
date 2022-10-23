import os
from functools import partial
from math import ceil

import jax.numpy as jnp
from jax import random as rnd
from jax import jit, vmap, pmap

from .vmc import create_energy_fn
from .utils import key_gen, split_variables_for_pmap, save_pk, load_pk
from .ansatz import create_wf


def create_sampler(mol, vwf, nan_safe=False):

    _step = step if not mol.pbc else partial(pbc_step, basis=mol.basis, inv_basis=mol.inv_basis)

    _sampler = partial(sample_metropolis_hastings, 
                        vwf=vwf, 
                        step_walkers=_step, 
                        correlation_length=mol.correlation_length, 
                        nan_safe=nan_safe,
                        target_acceptance=mol.target_acceptance)
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        _sampler = pmap(_sampler, in_axes=(None, 0, 0, 0))
        return _sampler

    if  bool(os.environ.get('DEBUG')) is True:
        return  _sampler

    return jit(_sampler)


def to_prob(amplitudes, nan_safe=False):
    ''' converts log amplitudes to probabilities '''
    ''' also catches nans (for the case the determinants are zero) '''
    probs = jnp.exp(amplitudes)**2
    if not nan_safe:
        return probs
    else:
        probs = jnp.where(jnp.isnan(amplitudes), 0.0, probs)
        probs = jnp.where(jnp.isinf(amplitudes), 0.0, probs)
        return probs


def step(walkers, key, shape, step_size):
    ''' takes a random step '''
    return walkers + rnd.normal(key, shape) * step_size


def pbc_step(walkers, key, shape, step_size, basis, inv_basis):
    ''' takes a random step, moves anything that stepped outside the simulation cell back inside '''
    ''' go to debugging/cell for more investigation '''
    walkers = walkers + rnd.normal(key, shape) * step_size
    walkers = keep_in_boundary(walkers, basis, inv_basis)
    return walkers


def transform_vector_space_sam(vectors: jnp.array, basis: jnp.array) -> jnp.array:
    '''
    case 1 catches non-orthorhombic cells 
    case 2 for orthorhombic and cubic cells
    '''
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis


def keep_in_boundary(walkers, basis, inv_basis):
    talkers = transform_vector_space_sam(walkers, inv_basis)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0., talkers + 1., talkers)
    talkers = transform_vector_space_sam(talkers, basis)
    return talkers


def sample_metropolis_hastings(params, 
                               curr_walkers, 
                               key, 
                               step_size, 
                               vwf,
                               step_walkers, 
                               correlation_length, 
                               nan_safe,
                               target_acceptance=0.5):

    shape = curr_walkers.shape

    amps = vwf(params, curr_walkers)
    curr_probs = to_prob(amps, nan_safe=nan_safe)

    acceptance_total = 0.

    # iterate with step size
    for _ in range(correlation_length//2):
        
        key, *subkeys = rnd.split(key, num=3)

        # next sample
        new_walkers = step_walkers(curr_walkers, subkeys[0], shape, step_size)
        new_probs = to_prob(vwf(params, new_walkers), nan_safe=nan_safe)

        # update sample
        alpha = new_probs / curr_probs
        mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

        curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
        curr_probs = jnp.where(mask_probs, new_probs, curr_probs)

        acceptance_total += mask_probs.mean()
        
    acceptance = acceptance_total / float(correlation_length//2)

    new_step_size = adjust_step_size(step_size, acceptance, key)

    acceptance_total = 0.

    # iterate with new step size
    for _ in range(correlation_length//2):
        
        key, *subkeys = rnd.split(key, num=3)

        # next sample
        new_walkers = step_walkers(curr_walkers, subkeys[0], shape, new_step_size)
        new_probs = to_prob(vwf(params, new_walkers), nan_safe=nan_safe)

        # update sample
        alpha = new_probs / curr_probs
        mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

        curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
        curr_probs = jnp.where(mask_probs, new_probs, curr_probs)

        acceptance_total += mask_probs.mean()
        
    new_acceptance = acceptance_total / float(correlation_length//2)

    step_size = jnp.where(jnp.abs(acceptance - target_acceptance) < jnp.abs(new_acceptance - target_acceptance), \
                                    step_size, new_step_size)

    return curr_walkers, (acceptance + new_acceptance) / 2, step_size


def metropolis_hastings_step(vwf, params, curr_walkers, curr_probs, key, shape, step_size, step_walkers):
    key, *subkeys = rnd.split(key, num=3)

    # next sample
    new_walkers = step_walkers(curr_walkers, subkeys[0], shape, step_size)
    new_probs = to_prob(vwf(params, new_walkers))

    # update sample
    alpha = new_probs / curr_probs
    mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

    curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
    curr_probs = jnp.where(mask_probs, new_probs, curr_probs)
    return curr_walkers, curr_probs, mask_probs


# def adjust_step_size(step_size, acceptance, target_acceptance=0.5):
#     decrease = ((acceptance < target_acceptance).astype(acceptance.dtype) * -2.) + 1.  # +1 for false, -1 for true
#     delta = decrease * 1. / 10000.
#     return jnp.clip(step_size + delta, 0.005, 0.2)


def adjust_step_size(step_size, acceptance, key, target_acceptance=0.5, std=0.001):
    step_size += std * rnd.normal(key, step_size.shape)
    return jnp.clip(step_size, 0.001, 1.)


def adjust_step_size_with_controlflow(step_size, acceptance, target_acceptance=0.5):
    if acceptance < target_acceptance:
        step_size -= 0.001
    else:
        step_size += 0.001
    return step_size



def initialise_walkers(mol,
                       vwf, 
                       sampler, 
                       params, 
                       keys,
                       walkers = None):

    if walkers is None:
        walkers = generate_walkers(mol.n_el_atoms, mol.atom_positions, mol.n_walkers, mol.n_el)
    
    elif not len(walkers) == mol.n_walkers:
        n_replicate = ceil(mol.n_walkers / len(walkers))
        walkers = jnp.concatenate([walkers for i in range(n_replicate)], axis=0)
        walkers = walkers[:mol.n_walkers, ...]

    if bool(os.environ.get('DISTRIBUTE')) is True:
        walkers = walkers.reshape(mol.n_devices, -1, *walkers.shape[1:])

    if mol.pbc:
        sampler = create_sampler(mol, vwf, nan_safe=True)
        print('sampling no infs, this could take a while')
        walkers = keep_in_boundary(walkers, mol.basis, mol.inv_basis)
        walkers = sample_until_no_infs(vwf, sampler, params, walkers, keys, mol.step_size)
        print('end sampling no infs')

    return walkers



def generate_walkers(n_el_atoms, atom_positions, n_walkers, n_el):
    """ Initialises walkers for pretraining

        Usage:
            walkers = initialize_walkers(ne_atoms, atom_positions, n_walkers).to(device=self.device, dtype=self.dtype)

        Args:
            ne_atoms (list int): number of electrons assigned to each nucleus
            atom_positions (list np.array): positions of the nuclei
            n_walkers (int): number of walkers

        Returns:
            walkers (np.array): walker positions (n_walkers, n_el, 3)

        """
    key = rnd.PRNGKey(1)
    if n_el_atoms is None:
        walkers = rnd.uniform(key, (n_walkers, n_el, 3))
    
    else:
        ups = []
        downs = []
        idx = 0
        for ne_atom, atom_position in zip(n_el_atoms, atom_positions):
            for e in range(ne_atom):
                key, *subkeys = rnd.split(key, num=3)

                if idx % 2 == 0:  # fill up the orbitals alternating up down
                    curr_sample_up = rnd.normal(subkeys[0], (n_walkers, 1, 3)) + atom_position
                    ups.append(curr_sample_up)
                else:
                    curr_sample_down = rnd.normal(subkeys[1], (n_walkers, 1, 3)) + atom_position
                    downs.append(curr_sample_down)

                idx += 1

        ups = jnp.concatenate(ups, axis=1)
        if len(downs) != 0:
            downs = jnp.concatenate(downs, axis=1)
            walkers = jnp.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
        else:
            walkers = ups
    return walkers


def sample_until_no_infs(vwf, sampler, params, walkers, keys, step_size):
    '''
    step_size: float
    '''

    n_walkers = walkers.shape[1]

    if bool(os.environ.get('DISTRIBUTE')) is True:
        vwf = pmap(vwf, in_axes=(None, 0))
    
    infs = True
    it = 0
    equilibration = 0
    while infs and equilibration < 10:
        # step_size += float(np.random.normal())
        keys, subkeys = key_gen(keys)

        walkers, acc, step_size = sampler(params, walkers, subkeys, step_size)
        log_psi = vwf(params, walkers)
        infs = jnp.isinf(log_psi).any() or jnp.isnan(log_psi).any()
        n_infs = jnp.sum(jnp.isinf(log_psi)).astype(float)
        n_nans = jnp.sum(jnp.isnan(log_psi)).astype(float)
        if not infs:
            equilibration += 1
        it += 1 
        if it % 10 == 0:
            print('step %i: %.2f %% infs and %.2f %% nans' % (it, n_infs/float(n_walkers), n_nans/float(n_walkers)))

    return walkers


def equilibrate(params, 
                walkers, 
                keys, 
                mol=None, 
                vwf=None, 
                sampler=None, 
                compute_energy=False, 
                n_it=1000, 
                step_size=0.02,
                step_size_out=False,
                walkers_path: str='./this_directory_does_not_exist'):
    
    if (not os.path.exists(walkers_path)) or (walkers_path == './this_directory_does_not_exist'):
        step_size = split_variables_for_pmap(walkers.shape[0], step_size)
        print('Equilibration')
        if sampler is None:
            if vwf is None:
                vwf = create_wf(mol)
            sampler = create_sampler(mol, vwf)
        if compute_energy:
            if vwf is None:
                vwf = create_wf(mol)
            compute_energy = create_energy_fn(mol, vwf)
            compute_energy = pmap(compute_energy, in_axes=(None, 0)) if bool(os.environ.get('DISTRIBUTE')) is True else compute_energy

        for it in range(n_it):
            keys, subkeys = key_gen(keys)

            walkers, acc, step_size = sampler(params, walkers, subkeys, step_size)
            
            if compute_energy:
                if it % 1000 == 0:
                    e_locs = compute_energy(params, walkers)
                    print(f'It {it} | Energy {jnp.mean(e_locs):.4f} | Acc {acc:.2f}')
        save_pk(walkers, walkers_path)
    else:
        print(f'Loading walkers {walkers_path}')
        walkers = load_pk(walkers_path)

        for it in range(100):
            keys, subkeys = key_gen(keys)
            walkers, acc, step_size = sampler(params, walkers, subkeys, step_size)

        step_size = split_variables_for_pmap(walkers.shape[0], step_size)

    if step_size_out is True:
        return walkers, step_size

    return walkers