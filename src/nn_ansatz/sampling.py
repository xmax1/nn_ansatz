
import numpy as np
import os

import jax
import jax.numpy as jnp
from jax import random as rnd
from jax import jit, vmap, pmap

from .vmc import create_energy_fn
from .utils import key_gen, split_variables_for_pmap


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
                   vwf,
                   mol,
                   correlation_length: int=10,
                   **kwargs):

    step = create_step(mol)

    def _body_fn(i, args):
        params, d0s, curr_walkers, curr_probs, shape, step_size, key, acceptance_total = args
        
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
        return params, d0s, curr_walkers, curr_probs, shape, step_size, key, acceptance_total

    # args = params, d0s, curr_walkers, curr_probs, shape, step_size, key, acceptance_total
    # args = jax.lax.for_iloop(0, correlation_length, _body_fn, args)
    # curr_walkers, acceptance_total, step_size = args[2], args[7], args[5]

    def _sample_metropolis_hastings(params, curr_walkers, d0s, key, step_size):

        shape = curr_walkers.shape
        curr_probs = to_prob(vwf(params, curr_walkers, d0s))

        acceptance_total = 0.

        # args = params, d0s, curr_walkers, curr_probs, shape, step_size, key, acceptance_total
        # args = jax.lax.fori_loop(0, correlation_length, _body_fn, args)
        # curr_walkers, acceptance_total, step_size = args[2], args[7], args[5]

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

        step_size = adjust_step_size(step_size, acceptance)

        return curr_walkers, acceptance, step_size

    if not os.environ.get('no_JIT') == 'True':
        _sample_metropolis_hastings = jit(_sample_metropolis_hastings)
    sampler = pmap(_sample_metropolis_hastings, in_axes=(None, 0, 0, 0, 0))

    compute_energy = pmap(create_energy_fn(wf, mol), in_axes=(None, 0, 0))

    def equilibrate(params, walkers, d0s, keys, n_it=1000, step_size=0.02):

        step_size = split_variables_for_pmap(walkers.shape[0], step_size)

        for i in range(n_it):
            keys, subkeys = key_gen(keys)

            walkers, acc, step_size = sampler(params, walkers, d0s, subkeys, step_size)
            e_locs = compute_energy(params, walkers, d0s)
            e_locs = jax.device_put(e_locs, jax.devices()[0]).mean()
            print('step %i energy %.4f acceptance %.2f' % (i, jnp.mean(e_locs), acc[0]))

        return walkers

    # if os.environ.get('JIT') == 'True':
        # return jit(sampler), equilibrate
    return sampler, equilibrate


def adjust_step_size(step_size, acceptance, target_acceptance=0.5):
    # if acceptance is larger ratio is 0.001 if smaller is zero
    scale = 1000.
    ratio = ((jnp.floor(target_acceptance / acceptance) * -1.) + 1.) / scale
    delta_acceptance = 1. / (2 * scale)
    step_change = ratio - delta_acceptance
    # step change is 0.0005 when
    return step_size + step_change
    # if acceptance < target_acceptance:
    #     step_size -= 0.001
    # else:
    #     step_size += 0.001
    # return step_size