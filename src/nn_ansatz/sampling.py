
import numpy as np
import jax.numpy as jnp
from jax import random as rnd
from jax import jit, vmap

from .vmc import create_energy_function


def to_prob(amplitudes):
    return jnp.exp(amplitudes)**2


def initialize_samples(ne_atoms, atom_positions, n_samples):
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                ups.append(curr_sample_up)
            else:
                curr_sample_down = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                downs.append(curr_sample_down)
    ups = np.concatenate(ups, axis=1)
    downs = np.concatenate(downs, axis=1)
    curr_sample = np.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return curr_sample


def create_sampler(wf, mol, correlation_length: int=10):

    vwf = vmap(wf, in_axes=(None, 0))

    def _sample_metropolis_hastings_loop(params, curr_walkers, key, step_size):

        shape = curr_walkers.shape
        curr_probs = to_prob(vwf(params, curr_walkers))

        acceptance_total = 0.
        for _ in range(correlation_length):
            key, *subkeys = rnd.split(key, num=3)

            # next sample
            new_walkers = curr_walkers + rnd.normal(subkeys[0], shape) * step_size
            new_probs = to_prob(vwf(params, new_walkers))

            # update sample
            alpha = new_probs / curr_probs
            mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

            curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
            curr_probs = jnp.where(mask_probs, new_probs, curr_probs)

            acceptance_total += mask_probs.mean()

        acceptance = acceptance_total / float(correlation_length)

        return curr_walkers, acceptance, step_size

    jit_sample_metropolis_hastings_loop = jit(_sample_metropolis_hastings_loop)

    def _sample_metropolis_hastings(params, walkers, key, step_size):
        """
        Notes:
            - This has to be done this way because adjust_step_size contains control flow
        """

        walkers, acceptance, step_size = jit_sample_metropolis_hastings_loop(params, walkers, key, step_size)
        step_size = adjust_step_size(step_size, acceptance)

        return walkers, acceptance, step_size

    compute_energy = create_energy_function(wf, mol)

    def _equilibrate(params, walkers, key, n_it=1000, step_size=0.02 ** 2):

        for i in range(n_it):
            key, subkey = rnd.split(key)
            e_locs = compute_energy(params, walkers)
            walkers, acc, step_size = _sample_metropolis_hastings(params, walkers, key, step_size)

            print('step %i energy %.4f acceptance %.2f' % (i, jnp.mean(e_locs), acc))

        return walkers

    return _sample_metropolis_hastings, _equilibrate


def adjust_step_size(step_size, acceptance, target_acceptance=0.5):
    if acceptance < target_acceptance:
        step_size -= 0.001
    else:
        step_size += 0.001
    return step_size



