import os
from functools import partial
from math import ceil

from jax import numpy as jnp
from jax import random as rnd
from jax import jit, vmap, pmap

from .vmc import create_energy_fn
from .utils import key_gen, split_variables_for_pmap, save_pk, load_pk
from .ansatz import create_wf



def pbc_step(walkers, key, shape, step_size, basis, inv_basis):
    ''' takes a random step, moves anything that stepped outside the simulation cell back inside '''
    ''' go to debugging/cell for more investigation '''
    walkers = walkers + rnd.normal(key, shape) * step_size
    walkers = keep_in_boundary(walkers, basis, inv_basis)
    return walkers


def create_sampler(mol, vwf):

    # walkers, key, shape, step_size, basis, inv_basis
    if mol.pbc:
        _pbc_step = partial(pbc_step, basis=mol.basis, inv_basis=mol.inv_basis)
        _sampler = partial(sample_metropolis_hastings, 
                           vwf=vwf, 
                           step_walkers=_pbc_step,
                           correlation_length=mol.correlation_length,
                           )
    else:
        _sampler = partial(sample_metropolis_hastings, 
                           vwf=vwf, 
                           step_walkers=step, 
                           correlation_length=mol.correlation_length
                           )

    if bool(os.environ.get('DISTRIBUTE')) is True:
        print('DISTRIBUTING SAMPLER')
        return pmap(_sampler, in_axes=(None, 0, 0, 0))

    return jit(_sampler)



def sample_metropolis_hastings(
    params, 
    curr_w, 
    key,
    step_size, 
    vwf,
    correlation_length,
    step_walkers, 
    step_size_std=0.001,
    target_acc = 0.5
    ):

    def c_prob(log_psi):
        return jnp.exp(log_psi)**2

    curr_p = c_prob(vwf(params, curr_w))

    acc = []
    for _ in range(correlation_length//2):
        key, subkey = rnd.split(key)
        new_w = step_walkers(curr_w, subkey, curr_w.shape, step_size)
        new_p = c_prob(vwf(params, new_w))

        key, subkey = rnd.split(key)
        mask_p = (new_p / curr_p) > rnd.uniform(subkey, new_p.shape)
        curr_w = jnp.where(mask_p[:, None, None], new_w, curr_w)
        curr_p = jnp.where(mask_p, new_p, curr_p)
        
        acc += [jnp.mean(mask_p)]
    acc = jnp.array(acc).mean()

    step_size_new = jnp.clip(step_size + step_size_std*rnd.normal(subkey), a_min=0.0001, a_max=1.)

    acc_new = []
    for _ in range(correlation_length//2):
        key, subkey = rnd.split(key)
        new_w = step_walkers(curr_w, subkey, curr_w.shape, step_size_new)
        new_p = c_prob(vwf(params, new_w))

        key, subkey = rnd.split(key)
        mask_p = (new_p / curr_p) > rnd.uniform(subkey, new_p.shape)
        curr_w = jnp.where(mask_p[:, None, None], new_w, curr_w)
        curr_p = jnp.where(mask_p, new_p, curr_p)
        
        acc_new += [jnp.mean(mask_p)]
    acc_new = jnp.array(acc_new).mean()

    
    mask = jnp.array((target_acc-acc)**2 < (target_acc-acc_new)**2, dtype=jnp.float32)
    not_mask = ((mask-1.)*-1.)
    step_size = mask*step_size + not_mask*step_size_new

    return curr_w, (acc+acc_new)/2., step_size


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


def sample_metropolis_hastings_skippy(params, 
                               curr_w, 
                               key, 
                               step_size, 
                               vwf,
                               step_walkers, 
                               correlation_length):
                               
    adjust = [0.001, 0.01, 0.1]
    tar_acc = 0.5

    curr_p = jnp.exp(vwf(params, curr_w))**2
    
    key, subkey = rnd.split(key)
    new_w = step_walkers(curr_w, subkey, curr_w.shape, out_stsi:=step_size)
    new_p = jnp.exp(vwf(params, new_w))**2
    
    key, subkey = rnd.split(key)
    mask_p = (new_p / curr_p) > rnd.uniform(subkey, new_p.shape)
    
    out_acc = jnp.mean(mask_p)
    curr_w = jnp.where(mask_p[:, None, None], new_w, curr_w)
    curr_p = jnp.where(mask_p, new_p, curr_p)

    for std in adjust:
        key, subkey = rnd.split(key)
        new_stsi = jnp.clip(step_size + std*rnd.normal(subkey), a_min=0.001, a_max=0.4)

        for step in range(len(adjust)):
            key, subkey = rnd.split(key)
            new_w = step_walkers(curr_w, subkey, curr_w.shape, new_stsi)
            new_p = jnp.exp(vwf(params, new_w))**2

            key, subkey = rnd.split(key)
            mask_p = (new_p / curr_p) > rnd.uniform(subkey, new_p.shape)
            curr_w = jnp.where(mask_p[:, None, None], new_w, curr_w)
            curr_p = jnp.where(mask_p, new_p, curr_p)

            new_acc = jnp.mean(mask_p)

        mask_stsi = (jnp.abs(out_acc - tar_acc) < jnp.abs(new_acc - tar_acc)).astype(float)  # float(True) = 1
        out_acc = mask_stsi*out_acc + jnp.abs(mask_stsi-1.)*new_acc
        out_stsi = mask_stsi*out_stsi + jnp.abs(mask_stsi-1.)*new_stsi

    acc = 0.0
    for step in range(correlation_length):
        key, subkey = rnd.split(key)
        new_w = step_walkers(curr_w, subkey, curr_w.shape, out_stsi)
        new_p = jnp.exp(vwf(params, new_w))**2

        key, subkey = rnd.split(key)
        mask_p = (new_p / curr_p) > rnd.uniform(subkey, new_p.shape)
        curr_w = jnp.where(mask_p[:, None, None], new_w, curr_w)
        curr_p = jnp.where(mask_p, new_p, curr_p)

        acc += jnp.mean(mask_p)

    return curr_w, acc / correlation_length, out_stsi




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
        sampler = create_sampler(mol, vwf)
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