
import numpy as np
import jax.numpy as jnp
from jax import random as rnd
from jax import jit, vmap

from .vmc import create_energy_fn


def to_prob(amplitudes):
    return jnp.exp(amplitudes)**2


def create_step(mol):

    def _no_boundaries_step(walkers, key, shape, step_size):
        return walkers + rnd.normal(key, shape) * step_size

    def _periodic_boundaries_step(walkers, key, shape, step_size):
        walkers = walkers + rnd.normal(key, shape) * step_size
        talkers = walkers.dot(mol.inv_real_basis)
        talkers = jnp.fmod(talkers, 1.)
        talkers = jnp.where(talkers < 0, talkers + 1., talkers)
        talkers = talkers.dot(mol.real_basis)
        return talkers

    if mol.periodic_boundaries:
        return _periodic_boundaries_step
    return _no_boundaries_step


def create_sampler(wf,
                   mol,
                   correlation_length: int=10):

    step = create_step(mol)

    vwf = vmap(wf, in_axes=(None, 0, 0))

    def _sample_metropolis_hastings_loop(params, curr_walkers, d0s, key, step_size):

        shape = curr_walkers.shape
        curr_probs = to_prob(vwf(params, curr_walkers, d0s))

        acceptance_total = 0.
        for _ in range(correlation_length):
            key, *subkeys = rnd.split(key, num=3)

            # next sample
            new_walkers = step(curr_walkers, subkeys[0], shape, step_size)
            new_probs = to_prob(vwf(params, new_walkers, d0s))

            # update sample
            alpha = new_probs / curr_probs
            mask_probs = alpha > rnd.uniform(subkeys[1], (shape[0],))

            curr_walkers = jnp.where(mask_probs[:, None, None], new_walkers, curr_walkers)
            curr_probs = jnp.where(mask_probs, new_probs, curr_probs)

            acceptance_total += mask_probs.mean()

        acceptance = acceptance_total / float(correlation_length)

        return curr_walkers, acceptance, step_size

    jit_sample_metropolis_hastings_loop = jit(_sample_metropolis_hastings_loop)

    def _sample_metropolis_hastings(params, walkers, d0s, key, step_size):
        """
        Notes:
            - This has to be done this way because adjust_step_size contains control flow
        """

        walkers, acceptance, step_size = jit_sample_metropolis_hastings_loop(params, walkers, d0s, key, step_size)
        step_size = adjust_step_size(step_size, acceptance)

        return walkers, acceptance, step_size

    compute_energy = create_energy_fn(wf, mol)

    def _equilibrate(params, walkers, d0s, key, n_it=1000, step_size=0.02 ** 2):

        for i in range(n_it):
            key, subkey = rnd.split(key)
            e_locs = compute_energy(params, walkers, d0s)
            walkers, acc, step_size = _sample_metropolis_hastings(params, walkers, d0s, subkey, step_size)

            print('step %i energy %.4f acceptance %.2f' % (i, jnp.mean(e_locs), acc))

        return walkers

    return _sample_metropolis_hastings, _equilibrate


def adjust_step_size(step_size, acceptance, target_acceptance=0.5):
    if acceptance < target_acceptance:
        step_size -= 0.001
    else:
        step_size += 0.001
    return step_size