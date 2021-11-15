import os
from collections import OrderedDict

import jax
from jax._src.tree_util import tree_flatten
import jax.random as rnd
import jax.numpy as jnp
from jax import jit, pmap, vmap
from jax.experimental.optimizers import adam
from tqdm import trange
import sys
import numpy as np

from .ansatz_base import apply_minimum_image_convention, drop_diagonal_i, split_and_squeeze

from .python_helpers import save_pk, update_dict

from .sampling import create_sampler, initialise_walkers, equilibrate
from .ansatz import create_wf
from .parameters import initialise_params
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import compute_potential_energy_i, create_energy_fn, create_grad_function, create_potential_energy
from .optimisers import kfac
from .utils import Logging, compare, load_pk, key_gen, save_config_csv_and_pickle, split_variables_for_pmap, write_summary_to_cfg


def check_inf_nan(arg):
    if type(arg) is OrderedDict:
        tree, _ = tree_flatten(arg)
        check = False
        for tree_arg in tree:
            check_tmp = jnp.isnan(tree_arg).any() or jnp.isinf(tree_arg).any()
            check = check or check_tmp
    else:
        check = jnp.isnan(arg).any() or jnp.isinf(arg).any()
    return check


def check_and_save(args, names):
    for arg in args:
        check = check_inf_nan(arg)
        if check:
            for arg, name in zip(args, names):
                save_pk(arg, '%s.pk' % name)
            sys.exit('found nans')


def initialise_system_wf_and_sampler(cfg, walkers=None):
    keys = rnd.PRNGKey(cfg['seed'])
    if bool(os.environ.get('DISTRIBUTE')) is True:
        keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)

    vwf = create_wf(mol)
    params = initialise_params(mol, keys)

    sampler = create_sampler(mol, vwf)

    if walkers is None:
        walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=walkers)

    if os.path.exists(cfg['pre_path']):
        print('loading pretrain wf %s' % cfg['pre_path'])
        params, walkers = load_pk(cfg['pre_path'])
        cfg['pretrain'] = False

    if cfg['pretrain']:
        params, walkers = pretrain_wf(mol, params, keys, sampler, walkers, **cfg)
    
    if cfg['load_it'] > 0:
        load_path = os.path.join(cfg['models_dir'], 'i%i.pk' % cfg['load_it'])
        if not os.path.exists(load_path):
            files = os.listdir(cfg['models_dir'])
            load_it = max([int(x.split('.')[0][1:]) for x in files])
            load_path = os.path.join(cfg['models_dir'], 'i%i.pk' % load_it)
        params = load_pk(load_path)

    
    return mol, vwf, walkers, params, sampler, keys

def run_vmc(cfg, walkers=None):

    logger = Logging(**cfg)

    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    grad_fn = create_grad_function(mol, vwf)

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])
    elif cfg['opt'] == 'adam':
        init, update, get_params = adam(cfg['lr'])
        update = jit(update)
        state = init(params)
    else:
        exit('Optimiser not available')

    steps = trange(1, cfg['n_it']+1, initial=1, total=cfg['n_it']+1, desc='training', disable=None)
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])

    for step in steps:
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        
        grads, e_locs = grad_fn(params, walkers)

        if cfg['opt'] == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers)

        state = update(step, grads, state)
        params = get_params(state)

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.log(step,
                   opt_state=state,
                   params=params,
                   e_locs=e_locs,
                   acceptance=acceptance)

    write_summary_to_cfg(cfg, logger.summary)
    logger.walkers = walkers
    
    return logger


def run_vmc_debug(cfg, walkers=None):

    logger = Logging(**cfg)

    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    grad_fn = create_grad_function(mol, vwf)
    e_fn = pmap(create_energy_fn(mol, vwf, separate=True), in_axes=(None, 0))
    pwf = pmap(vwf, in_axes=(None, 0))

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])
    elif cfg['opt'] == 'adam':
        init, update, get_params = adam(cfg['lr'])
        update = jit(update)
        state = init(params)
    else:
        exit('Optimiser not available')

    steps = trange(1, cfg['n_it']+1, initial=1, total=cfg['n_it']+1, desc='training', disable=None)
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])

    for step in steps:
        keys, subkeys = key_gen(keys)
        
        logpsi = pwf(params, walkers)
        probs = jnp.exp(logpsi)**2
        grads, e_locs = grad_fn(params, walkers)
        pe, ke = e_fn(params, walkers)

        grads_check = check_inf_nan(grads)

        if cfg['opt'] == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers)

        walkers_check = check_inf_nan(walkers)
        pe_check = check_inf_nan(pe)
        ke_check = check_inf_nan(ke)
        prob_check = check_inf_nan(probs)
        # kfac_grads_check = check_inf_nan(grads)

        if walkers_check or grads_check or pe_check or ke_check or prob_check:
            print('nans in the house')
            return walkers, grads, pe, ke, probs

        state = update(step, grads, state)
        params = get_params(state)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()
    
    return walkers, grads, pe, ke, probs


def approximate_value(function, params, sampler, walkers, keys, names, cfg, n_it=1000):
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])
    values = {}
    for i in range(n_it):
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)

        new_values = function(params, walkers)
        if type(new_values) is tuple:
            for name, new_value in zip(names, new_values):
                new_value = jnp.mean(new_value)
                update_dict(values, name, new_value)
        else:
            new_value = jnp.mean(new_values)
            update_dict(values, names[0], new_value)


        if i % 100 == 0:
            print('step %i ' % i)
    
    return values


def approximate_ke_pe(cfg, load_it=None, n_it=1000, walkers=None):
    if load_it is not None:
        cfg['load_it'] = load_it
    names = ['pe', 'ke']
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)
    energy_function = create_energy_fn(mol, vwf, separate=True)
    values = approximate_value(energy_function, params, sampler, walkers, keys, names, cfg, n_it=n_it)
    return values


def compute_per_particle(values, n_particles):
    new_dict = {}
    for k, v in values.items():
        new_dict[k] = v
        new_dict[k + '_per_particle'] = v / float(n_particles)
    return new_dict

def approximate_energy(cfg, load_it=None, n_it=10000, walkers=None):
    if load_it is not None:
        cfg['load_it'] = load_it

    os.environ['DISTRIBUTE'] = 'True'
    
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)
    walkers = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=True, n_it=1000)
    energy_function = create_energy_fn(mol, vwf, separate=True)
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        energy_function = pmap(energy_function, in_axes=(None, 0))
    
    names = ['pe', 'ke']
    values = approximate_value(energy_function, params, sampler, walkers, keys, names, cfg, n_it=n_it)
    
    n_particles = mol.n_atoms if mol.n_atoms != 0 else mol.n_el
    n_samples = len((values[names[0]]))
    
    save_values = {}
    save_values['equilibrated_pe_mean_i%i' % load_it] = np.mean(values['pe'])
    save_values['equilibrate_pe_std_i%i' % load_it] = np.std(values['pe'])
    save_values['equilibrate_pe_sem_i%i' % load_it] = np.std(values['pe']) / np.sqrt(n_samples)

    save_values['equilibrate_ke_mean_i%i' % load_it] = np.mean(values['ke'])
    save_values['equilibrate_ke_std_i%i' % load_it] = np.std(values['ke'])
    save_values['equilibrate_ke_sem_i%i' % load_it] = np.std(values['ke']) / np.sqrt(n_samples)

    energies = np.array(values['pe']) + np.array(values['ke'])
    save_values['equilibrated_energy_mean_i%i' % load_it] = np.mean(energies)
    save_values['equilibrated_energy_std_i%i' % load_it] = np.std(energies)
    save_values['equilibrated_energy_sem_i%i' % load_it] = np.std(energies) / np.sqrt(n_samples)

    # save_values = compute_per_particle(save_values, n_particles)
    cfg.update(save_values)

    save_config_csv_and_pickle(cfg)
    return values


def approximate_pair_distribution_function(cfg, load_it=None, n_bins=100, n_it=1000, walkers=None):
    n_bins += 1
    
    if load_it is not None:
        cfg['load_it'] = load_it

    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)
    n_walkers, n_el = walkers.shape[:2]
    bins = np.linspace(0., mol.scale_cell/2., n_bins)
    print(bins)
    
    walkers = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=True, n_it=200)

    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])
    drop_diagonal = vmap(drop_diagonal_i, in_axes=(0,))

    binned_ee_distances_list = []
    for i in range(n_it):
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        ee_vectors = walkers[:, None, ...] - walkers[:, :, None, ...]
        ee_vectors = apply_minimum_image_convention(ee_vectors, mol.basis, mol.inv_basis)
        # ee_vectors = ee_vectors[:, 0, 1:, :]
        ee_vectors = drop_diagonal(ee_vectors)  # (n_walkers, n_el - 1, 3)

        ee_distances = np.array(jnp.linalg.norm(ee_vectors, axis=-1))
        binned_dists = np.digitize(ee_distances, bins=bins).reshape(-1) # divide by r**2
        bin_freqs = (np.bincount(binned_dists, minlength=(n_bins+1)) / (float(n_walkers)))[:-1]

        binned_ee_distances_list.append(bin_freqs)

        if i % 10 == 0:
            print('step %i ' % i)
    
    binned_ee_distances = jnp.array(binned_ee_distances_list)
    mean = jnp.mean(jnp.array(binned_ee_distances), axis=0)
    sem = jnp.std(jnp.array(binned_ee_distances), axis=0) / jnp.sqrt(len(binned_ee_distances))
    return mean, sem

# np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=m)

def compare_einsum(cfg, walkers=None):
    logger = Logging(**cfg)

    keys = rnd.PRNGKey(cfg['seed'])
    if bool(os.environ.get('DISTRIBUTE')) is True:
        keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)
    mol_einsum = SystemAnsatz(**cfg)

    vwf = create_wf(mol, distribute=True)
    params = initialise_params(mol, keys)

    sampler = create_sampler(mol, vwf)
    
    mol_einsum.einsum = True
    vwf_einsum = create_wf(mol_einsum, distribute=True)
    params_einsum = initialise_params(mol_einsum, keys)
    
    compare_params(params, params_einsum)
        
    if walkers is None:
        walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=walkers)

    vwf = pmap(vwf, in_axes=(None, 0))
    vwf_einsum = pmap(vwf_einsum, in_axes=(None, 0))
    log_psi = vwf(params, walkers)
    log_psi_einsum = vwf_einsum(params_einsum, walkers)

    difference = jnp.mean(log_psi - log_psi_einsum)
    print('difference = %.2f' % difference)


def compare_params(params, params_einsum):
    pi_ups = [split_and_squeeze(p, axis=2) for p in split_and_squeeze(params_einsum['env_pi_up'], axis=2)]
    pi_ups = [x for y in pi_ups for x in y]

    pi_downs = [split_and_squeeze(p, axis=2) for p in split_and_squeeze(params_einsum['env_pi_down'], axis=2)]
    pi_downs = [x for y in pi_downs for x in y]

    for dictionary, name in zip([pi_ups, pi_downs], ['env_pi_up', 'env_pi_down']):
        for i, v in enumerate(dictionary):
            params_einsum[name + '_%i' %i] = v
    
    params_einsum = {k:v for k, v in params_einsum.items() if not 'env_pi_up' == k}
    params_einsum = {k:v for k, v in params_einsum.items() if not 'env_pi_down' == k}

    for (k1, v1), (k2, v2) in zip(params.items(), params_einsum.items()):
        if not v1.shape == v2.shape:
            print(v1.shape, v2.shape)
            v2 = v2.reshape(v1.shape)
        print(jnp.mean(v1 - v2))


if __name__ == '__main__':
    import sys
    sys.path.append('.')

    from utils import setup
    cfg = setup(system='LiSolidBCC',
                    n_pre_it=0,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=64,
                    step_size=0.05,
                    n_ph=32,
                    scalar_inputs=False,
                    orbitals='anisotropic',
                    einsum=True,
                    n_periodic_input=1,
                    opt='adam',
                    n_det=4,
                    print_every=50,
                    save_every=2500,
                    lr=1e-4,
                    n_it=30000)

    compare_einsum(cfg)

