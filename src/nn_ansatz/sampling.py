
import numpy as np
import os
from functools import partial
import math 

import jax
import jax.numpy as jnp
from jax import random as rnd
from jax import jit, vmap, pmap

from .vmc import create_energy_fn
from .utils import key_gen, split_variables_for_pmap


def create_step(mol):

    if mol.periodic_boundaries:
        return partial(periodic_boundaries_step, real_basis=mol.real_basis, inv_real_basis=mol.inv_real_basis)
    
    return step


def create_sampler(mol, vwf):

    _step = create_step(mol)

    _sampler = partial(sample_metropolis_hastings, vwf=vwf, step_walkers=_step, correlation_length=mol.correlation_length)
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        _sampler = pmap(_sampler, in_axes=(None, 0, 0, 0))

    return jit(_sampler)


def to_prob(amplitudes):
    ''' converts log amplitudes to probabilities '''
    ''' also catches nans (for the case the determinants are zero) '''
    amplitudes = jnp.nan_to_num(amplitudes, nan=-1.79769313**308)
    return jnp.exp(amplitudes)**2


def step(walkers, key, shape, step_size):
    ''' takes a random step '''
    return walkers + rnd.normal(key, shape) * step_size


def periodic_boundaries_step(walkers, key, shape, step_size, real_basis, inv_real_basis):
    ''' takes a random step, moves anything that stepped outside the simulation cell back inside '''
    walkers = walkers + rnd.normal(key, shape) * step_size
    talkers = walkers.dot(inv_real_basis.T)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0, talkers + 1., talkers)
    talkers = talkers.dot(real_basis.T)
    return talkers


def sample_metropolis_hastings(params, curr_walkers, key, step_size, vwf, step_walkers, correlation_length):

    shape = curr_walkers.shape

    amps = vwf(params, curr_walkers)
    curr_probs = to_prob(amps)

    acceptance_total = 0.

    for _ in range(correlation_length):
        
        key, *subkeys = rnd.split(key, num=3)

        # next sample
        new_walkers = step_walkers(curr_walkers, subkeys[0], shape, step_size)
        new_probs = to_prob(vwf(params, new_walkers))

        # update sample
        alpha = new_probs / curr_probs
        mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

        curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
        curr_probs = jnp.where(mask_probs, new_probs, curr_probs)

        acceptance_total += mask_probs.mean()
        
    acceptance = acceptance_total / float(correlation_length)

    step_size = adjust_step_size(step_size, acceptance)

    return curr_walkers, acceptance, step_size


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



def adjust_step_size(step_size, acceptance, target_acceptance=0.5):
    decrease = ((acceptance < target_acceptance).astype(acceptance.dtype) * -2.) + 1.  # +1 for false, -1 for true
    delta = decrease * 1. / 1000.
    return step_size + delta


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
        walkers = generate_walkers_around_nuclei(mol.n_el_atoms, mol.atom_positions, mol.n_walkers)
    elif not len(walkers) == mol.n_walkers:
        n_replicate = math.ceil(mol.n_walkers / len(walkers))
        walkers = jnp.concatenate([walkers for i in range(n_replicate)], axis=0)
        walkers = walkers[:mol.n_walkers, ...]

    if bool(os.environ.get('DISTRIBUTE')) is True:
        walkers = walkers.reshape(mol.n_devices, -1, *walkers.shape[1:])
    print('sampling no infs, this could take a while')
    walkers = sample_until_no_infs(vwf, sampler, params, walkers, keys, mol.step_size)
    print('end sampling no infs')

    return walkers



def generate_walkers_around_nuclei(ne_atoms, atom_positions, n_walkers):
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
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            key, subkey = rnd.split(key)

            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = rnd.normal(subkey, (n_walkers, 1, 3)) + atom_position
                ups.append(curr_sample_up)
            else:
                curr_sample_down = rnd.normal(subkey, (n_walkers, 1, 3)) + atom_position
                downs.append(curr_sample_down)

    ups = jnp.concatenate(ups, axis=1)
    downs = jnp.concatenate(downs, axis=1)
    curr_sample = jnp.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return curr_sample


def sample_until_no_infs(vwf, sampler, params, walkers, keys, step_size):

    step_size += 4.

    if bool(os.environ.get('DISTRIBUTE')) is True:
        vwf = pmap(vwf, in_axes=(None, 0))
    
    infs = True
    while infs:
        keys, subkeys = key_gen(keys)

        walkers, acc, step_size = sampler(params, walkers, subkeys, step_size)
        log_psi = vwf(params, walkers)
        infs = jnp.isinf(log_psi).any() or jnp.isnan(log_psi).any()

    return walkers

