import os
from collections import OrderedDict
from itertools import product 

import jax.random as rnd
import jax.numpy as jnp
from jax import jit, pmap, vmap
# from jax.experimental.optimizers import adam
from optax import adam, apply_updates
from tqdm import trange
import sys
import numpy as np
from jax.tree_util import tree_flatten
from jax import tree_util
import time
from pathlib import Path

from .ansatz_base import apply_minimum_image_convention, drop_diagonal_i, split_and_squeeze
from .python_helpers import save_pk, update_dict
from .sampling import create_sampler, initialise_walkers
from .ansatz import create_wf
from .parameters import initialise_params
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
from .optimisers import kfac
from .utils import Logging, load_pk, save_config_csv_and_pickle, split_variables_for_pmap


def key_gen(keys):
    if len(keys.shape) == 2:  # if distributed
        keys = jnp.array([rnd.split(key) for key in keys])
        keys = jnp.split(keys, 2, axis=1)
        return [x.squeeze(axis=1) for x in keys]
    return rnd.split(keys)

# def key_gen(keys):
#     if len(keys.shape) == 2:  # if distributed
#         return jnp.stack([rnd.split(key) for key in keys])
#         # keys = jnp.split(keys, 2, axis=1)
#         # return [x.squeeze(axis=1) for x in keys]
#     return rnd.split(keys)


def create_key(seed, n_devices):
    return rnd.split(rnd.PRNGKey(seed), n_devices).reshape(n_devices, 2), rnd.PRNGKey(seed)
    

def walker_init(n_walker, n_el, scale, n_device):
    walker_device = []
    shift = scale / 4.
    for _ in range(n_device):
        sites = np.linspace(0., scale, 4, endpoint=True)[:-1] + (shift/2.)
        sites_all = np.array(list(product(*[sites, sites, sites])))
        walker = []
        for _ in range(n_walker):
            idxs = np.random.choice(np.arange(len(sites_all)), size=n_el, replace=False)
            w = sites_all[idxs]
            walker += [w]   
        walker_device += [jnp.array(np.stack(walker, axis=0))]
    return jnp.stack(walker_device)


def equilibrate(
    key, 
    param, 
    walker, 
    sampler, 
    step_size=0.02, 
    n_eq=1000, 
    e_every=1, 
    compute_e=None
    ):

    eq_stat = {'pe': [], 'ke': [], 'e': [], 'walker': []}
    for step in range(n_eq):
        
        for _ in range(10):
            key, subkey = key_gen(key)
            walker, acc, step_size = sampler(param, walker, subkey, step_size)

        eq_stat['walker'] += [np.array(walker)]
        # print(jnp.mean(acc), jnp.mean(step_size), jnp.sum(jnp.isnan(walker)))
        # print((compute_e is not None), (step % e_every == 0), step, e_every)
        if (compute_e is not None) and (step % e_every == 0):
            pe, ke = compute_e(param, walker)
            e = pe+ke

            eq_stat['pe'] += [np.array(pe)]  # like this to save GPU memory
            eq_stat['ke'] += [np.array(ke)]
            eq_stat['e'] += [np.array(e)]
            
            print(f'Step: {step} / {n_eq} | E: {jnp.mean(e):.7f} +- {jnp.std(e):.7f}')
            print(f'Step_size: {float(jnp.mean(step_size)):.3f} | Acc.: {float(jnp.mean(acc)):.2f}')        

    if compute_e is not None:
        print(f'E: {jnp.mean(jnp.array(eq_stat["e"])):.7f} +- {jnp.std(jnp.array(eq_stat["e"])):.7f}')
    return {k: np.concatenate(v, axis=0) for k,v in eq_stat.items() if len(v) > 0}


def compute_energy_from_save(cfg):

    key, key_local = create_key(cfg['seed'], cfg['n_devices'])  # (1, 2)
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])  # (1,)

    mol = SystemAnsatz(**cfg)
    vwf = create_wf(mol)
    sampler = create_sampler(mol, vwf)
    compute_e = pmap(create_energy_fn(mol, vwf, separate=True), in_axes=(None, 0))
    load_it = cfg['load_it']

    print('BASIS')
    print(mol.basis)
    print(mol.inv_basis)

    param_path = Path(cfg['run_dir'], 'models', f'i{load_it}.pk')
    walker_path = Path(cfg['run_dir'], 'models', f'w_tr_i{load_it}.pk')

    print(f'LOADING PARAM: \n {param_path}')
    param = load_pk(param_path)

    if walker_path is not None:
        print(f'LOADING WALKER: \n {walker_path}')
        walker = load_pk(walker_path)[:cfg['n_devices']]
    else:
        walker = walker_init(cfg['n_walkers_per_device'], cfg['n_el'], mol.scale_cell, cfg['n_devices'])
        eq_stat = equilibrate(key, param, walker, sampler, step_size, n_eq=1000)

    if len(walker) > cfg['n_walkers_per_device']:
        walker = walker[:, :cfg['n_walkers_per_device']]

    n_batch = ((cfg['n_compute'] // cfg['n_walkers_per_device']) + 1)
    eq_stat = {'pe': [], 'ke': [], 'e': [], 'walker': []}
    for step in range(n_batch):
        for _ in range(10):
            key, subkey = key_gen(key)
            print(walker.shape, subkey.shape, step_size.shape)
            print(param.shape)
            walker, acc, step_size = sampler(param, walker, subkey, step_size)

        pe, ke = compute_e(param, walker)
        e = pe+ke

        eq_stat['pe'] += [np.array(pe)]  # like this to save GPU memory
        eq_stat['ke'] += [np.array(ke)]
        eq_stat['e'] += [np.array(e)]
        eq_stat['walker'] += [np.array(walker)]

        print(f'Step: {step} / {n_batch} | E: {jnp.mean(e):.7f} +- {jnp.std(e):.7f}')
        print(f'Step_size: {float(jnp.mean(step_size)):.3f} | Acc.: {float(jnp.mean(acc)):.2f}')        

    print(f'E: {jnp.mean(jnp.array(eq_stat["e"])):.7f} +- {jnp.std(jnp.array(eq_stat["e"])):.7f}')
    eq_stat = {k: np.concatenate(v, axis=0) for k,v in eq_stat.items()}
    save_pk(eq_stat, os.path.join(cfg['run_dir'], f'eq_stats_i{load_it}.pk'))
    return None


def transform_vector_space(vectors: jnp.array, basis: jnp.array) -> jnp.array:
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis


def keep_in_boundary(walkers, basis, inv_basis):   # some numerical errors
    talkers = transform_vector_space(walkers, inv_basis)
    talkers = jnp.fmod(talkers, 1.)  # y – The remainder of the division of x1 by x2.
    talkers = jnp.where(talkers < 0., talkers + 1., talkers)
    talkers = transform_vector_space(talkers, basis)
    return talkers


def run_vmc(cfg):

    for k, v in cfg.items():
        print(k, '\n', v, '\n')

    key, key_local = create_key(cfg['seed'], cfg['n_devices'])
    step_size = split_variables_for_pmap(cfg['n_devices'], cfg['step_size'])
    print(step_size)
        
    logger = Logging(**cfg)
    mol = SystemAnsatz(**cfg)
    vwf = create_wf(mol)
    sampler = create_sampler(mol, vwf)
    compute_e = pmap(create_energy_fn(mol, vwf, separate=True), in_axes=(None, 0))
    params = initialise_params(mol, key_local)
    walkers = walker_init(cfg['n_walkers_per_device'], cfg['n_el'], mol.scale_cell, cfg['n_devices'])
    walkers = keep_in_boundary(walkers, mol.basis, mol.inv_basis)
    walkers = equilibrate(key, params, walkers, sampler, step_size, n_eq=100)['walker']  # because list accumulates them and pmap stuff
    walkers = walkers[:cfg['n_devices']]
    print(walkers.shape)
    if walkers.ndim == 3:
        walkers = walkers[None, ...]

    print(walkers.shape)
    if cfg['pretrain']:
        params, walkers = pretrain_wf(mol, params, key, sampler, walkers, **cfg)  # OUTPUTS (n_walkers, ...)
        for _ in range(1000):
            key, subkey = key_gen(key)
            walkers, acceptance, step_size = sampler(params, walkers, subkey, step_size)

    print(walkers.shape)
    print(jnp.mean(step_size), step_size.shape)
    grad_fn = create_grad_function(mol, vwf)

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])
    elif cfg['opt'] == 'adam':
        optimizer = adam(cfg['lr'])
        state = optimizer.init(params)
    else:
        exit('Optimiser not available')

    steps = trange(1, cfg['n_it']+1, initial=1, total=cfg['n_it']+1, desc='%s/training' % cfg['exp_name'], disable=None)

    e_locs_cpu = []
    for step in steps:
        step_from_1 = step + 1
        key, subkey = key_gen(key)
        walkers, acceptance, step_size = sampler(params, walkers, subkey, step_size)
        grads, e_locs = grad_fn(params, walkers, jnp.array([step]))
        
        try:
            def save_us_from_nans(w):
                if jnp.sum(e_locs_nan := jnp.isnan(e_locs)):
                    print('SAVING US FROM NANS')
                    for i in range(10):
                        key, subkey = key_gen(key)
                        w, acceptance, step_size = sampler(params, w, subkey, step_size)
                    w = jnp.where(e_locs_nan, w, walkers)
                return w
            walkers = save_us_from_nans(walkers)
        except Exception as e:
            print(f'BUG INDA NAN {e}')

        # if (1.-(step/cfg['n_it'])) < cfg['final_sprint']:
        #     grads, tree = tree_util.tree_flatten(grads)
        #     for n_minus_1 in range(1, 10):
        #         key, subkey = key_gen(key)
        #         walkers, acceptance, step_size = sampler(params, walkers, subkey, step_size)
        #         grads_extra, e_locs_extra = grad_fn(params, walkers, jnp.array([step]))
        #         grads_extra, _ = tree_util.tree_flatten(grads_extra)
        #         grads = [g + g_e for g, g_e in zip(grads, grads_extra)]
        #         e_locs = jnp.concatenate([e_locs, e_locs_extra])
        #     grads = tree_util.tree_unflatten(tree, [g/n_minus_1 for g in grads])
        #     step += 1

        if cfg['opt'] == 'kfac':
            grads, state = kfac_update(step, grads, state, walkers)
            state = update(step, grads, state)
            params = get_params(state)
        else:
            grads, state = optimizer.update(grads, state)
            params = apply_updates(params, grads)

        steps.set_postfix(E=f'{jnp.mean(e_locs):.6f}')
        steps.refresh()

        logger.writer('w/step_size', float(jnp.mean(step_size)), step)
        logger.writer('w/acc', float(jnp.mean(acceptance)), step)
        logger.writer('w/e_mean', float(jnp.mean(e_locs)), step)
        logger.writer('w/e_std', float(jnp.std(e_locs)), step)

        if step_from_1 % 100 == 0:
            print(f'Step {step_from_1}: Step_size: {float(jnp.mean(step_size)):.3f} | Acc.: {float(jnp.mean(acceptance)):.2f}')

        e_locs_cpu += [np.array(e_locs)]
        if step_from_1 % cfg['save_every'] == 0:
            try:
                save_pk(params, os.path.join(cfg['models_dir'], 'i%i.pk' % step_from_1))
                save_pk(np.array(walkers), os.path.join(cfg['models_dir'], 'w_tr_i%i.pk' % step_from_1))
                save_pk(np.concatenate(e_locs_cpu), os.path.join(cfg['models_dir'], 'e_tr_i%i.pk' % step_from_1))
            except:
                print('SAVING DID NOT WORK')

        if jnp.isnan(jnp.mean(e_locs)):
            exit('found nans')

    print('WARMING UP EQ')
    n_eq_walkers = 5000000 if cfg['n_it'] > 500 else cfg['n_walkers'] * 40
    n_eq = int((n_eq_walkers//cfg['n_walkers'])) + 1
    eq_stats = {'pe': [], 'ke': [], 'e': [], 'walkers': []}

    for step_eq in range(n_eq):
        for _ in range(10):
            key, subkey = key_gen(key)
            walkers, acceptance, step_size = sampler(params, walkers, subkey, step_size)
        
        if step_eq % 100:
            pe, ke = compute_e(params, walkers)
            e = pe+ke
            eq_stats['pe'] += [np.array(pe)]
            eq_stats['ke'] += [np.array(ke)]
            eq_stats['e'] += [np.array(e)]
        
        print(f'Step: {step_eq} / {n_eq} | E: {jnp.mean(e):.7f} +- {jnp.std(e):.7f}')
        print(f'Step_size: {float(jnp.mean(step_size)):.4f} | Acc.: {float(jnp.mean(acceptance)):.2f}')
        
        eq_stats['walkers'] += [np.array(walkers)]

    print(f'E: {jnp.mean(jnp.array(eq_stats["e"])):.7f} +- {jnp.std(jnp.array(eq_stats["e"])):.7f}')
    eq_stats = {k: np.concatenate(v, axis=0) for k,v in eq_stats.items()}
    save_pk(eq_stats, os.path.join(cfg['run_dir'], f'eq_stats_i{step_eq}.pk'))
    return None



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


def confirm_antisymmetric(mol, params, walkers):
    swf = create_wf(mol, signed=True)
    idx = 1
    if bool(os.environ['DISTRIBUTE']) == True:
        idx += 1
        swf = pmap(swf, in_axes=(None, 0))

    bl, bs = swf(params, walkers)
    nw_up, nw_down = walkers.split([mol.n_up], axis=idx)
    
    idxs = [1, 0]
    idxs.extend(list(range(2, mol.n_up)))
    nw_up = nw_up[..., idxs, :]
    walkers = jnp.concatenate([nw_up, nw_down], axis=idx)
    tl1, ts1 = swf(params, walkers)

    up_swap_mean = float(jnp.abs(bl-tl1).mean())
    up_swap_smean =  float(jnp.abs(bs-ts1).mean())
    
    print('antisymmetry check:')
    print('swap ups || difference %.2f sign difference %.2f' % (up_swap_mean, up_swap_smean))

    
    if not mol.n_down == 0:
        idxs = [1, 0]
        idxs.extend(list(range(2, mol.n_down)))
        nw_down = nw_down[..., idxs, :]
        walkers = jnp.concatenate([nw_up, nw_down], axis=idx)
        tl2, ts2 = swf(params, walkers)

        down_swap_mean = float(jnp.abs(tl2-tl1).mean())
        down_swap_smean =  float(jnp.abs(ts2-ts1).mean())
        
        print('swap downs || difference %.2f sign difference %.2f' % (down_swap_mean, down_swap_smean))


def initialise_system_wf_and_sampler(cfg, params_path=None, walkers=None, walkers_path=None):
    keys = rnd.PRNGKey(cfg['seed'])
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

    mol = SystemAnsatz(**cfg)
    vwf = create_wf(mol)
    sampler = create_sampler(mol, vwf)

    if params_path is None:
        params = initialise_params(mol, keys)
    else:
        print(f'Loading params: \n {params_path}')
        params = load_pk(params_path)

    if walkers_path is None:
        if walkers is None:
            print('Initialising and equilibrating walkers')
            walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=walkers)
            walkers = equilibrate(params, walkers, keys, mol, vwf=vwf, sampler=sampler, n_it=1000)

    if (params_path is None) and cfg['pretrain']:
        params, walkers = pretrain_wf(mol, params, keys, sampler, walkers, **cfg)
    
    return mol, vwf, walkers, params, sampler, keys
    

def time_sample(cfg, walkers=None):
    
    print('executing measure time')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    step_size = get_step_size(walkers, params, sampler, keys)

    times = []
    for step in range(cfg['n_it']*10 + 10):
        t0 = time.time()
        keys, subkeys = key_gen(keys)
        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        t1 = time.time()
        times.append(t1 - t0)
    return times


def time_grads(cfg, walkers=None):
    
    print('executing measure time')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    # grad_fn = create_grad_function(mol, vwf)

    update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])

    energy_function = create_energy_fn(mol, vwf, separate=True)
    if bool(os.environ.get('DISTRIBUTE')) is True:
        energy_function = pmap(energy_function, in_axes=(None, 0))

    step_size = get_step_size(walkers, params, sampler, keys)

    ps, tree = tree_flatten(params)
    grads = [jnp.zeros(p.shape) for p in ps]

    energy_function = create_energy_fn(mol, vwf, separate=True)
    if bool(os.environ.get('DISTRIBUTE')) is True:
        energy_function = pmap(energy_function, in_axes=(None, 0))
    
    times = []
    for step in range(cfg['n_it']+10):

        t0 = time.time()

        keys, subkeys = key_gen(keys)
        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        # grads, e_locs = grad_fn(params, walkers, step)
        # pe, ke = energy_function(params, walkers)
        grads, state = kfac_update(step, grads, state, walkers)
        state = update(step, grads, state)
        params = get_params(state)

        t1 = time.time()
        times.append(t1 - t0)
    return times


def time_energy(cfg, walkers=None):

    print('executing measure time')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    energy_function = create_energy_fn(mol, vwf, separate=True)
    if bool(os.environ.get('DISTRIBUTE')) is True:
        energy_function = pmap(energy_function, in_axes=(None, 0))

    step_size = get_step_size(walkers, params, sampler, keys)

    times = []
    for i in range(cfg['n_it']+10):
        t0 = time.time()

        keys, subkeys = key_gen(keys)
        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        pe, ke = energy_function(params, walkers)

        t1 = time.time()
        times.append(t1 - t0)
    return times



def measure_kfac_and_energy_time(cfg, walkers=None):

    
    times = time_sample(cfg)
    cfg['sampling_time'] = np.mean(times[10:])

    # times = time_energy(cfg)
    # cfg['sampling_and_energy_time'] = np.mean(times[10:])
   
    # times = time_grads(cfg)
    # cfg['sampling_and_kfac_and_energy_time'] = np.mean(times[10:])

    times = time_grads(cfg)
    cfg['sampling_and_kfac_time'] = np.mean(times[10:])

    cfg['kfac_time'] = cfg['sampling_and_kfac_time'] - cfg['sampling_time']
    # cfg['energy_time'] = cfg['sampling_and_energy_time'] - cfg['sampling_time']
    # cfg['kfac_time'] = cfg['sampling_and_kfac_and_energy_time'] - cfg['sampling_time'] - cfg['energy_time']

    save_config_csv_and_pickle(cfg)
    
    return cfg


def run_vmc_debug(cfg, walkers=None):

    logger = Logging(**cfg)

    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers)

    grad_fn = create_grad_function(mol, vwf)
    e_fn = pmap(create_energy_fn(mol, vwf, separate=True), in_axes=(None, 0))
    pwf = pmap(vwf, in_axes=(None, 0))

    if cfg['opt'] == 'kfac':
        update, get_params, kfac_update, state = kfac(mol, params, walkers, cfg['lr'], cfg['damping'], cfg['norm_constraint'])
    elif cfg['opt'] == 'adam':
        # init, update, get_params = adam(cfg['lr'])
        # update = jit(update)
        # state = init(params)
        optimizer = adam(cfg['lr'])
        opt_state = optimizer.init(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
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
                new_value_std = jnp.std(new_value)
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


def get_step_size(walkers, params, sampler, keys):
    n_devices = walkers.shape[0]
    step_sizes = [0.5, 0.1, 0.05, 0.01, 0.005]
    acceptances = []
    for step_size in step_sizes:
        step_size = split_variables_for_pmap(n_devices, step_size)
        keys, subkeys = key_gen(keys)
        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)
        acceptance = float(jnp.mean(acceptance))
        acceptances.append(acceptance)
    acceptances = np.abs(np.array(acceptances) - 0.5)
    idx = np.argmin(acceptances)
    step_size = step_sizes[idx]
    return split_variables_for_pmap(n_devices, step_size)


oj = os.path.join 

def approximate_energy(cfg, run_dir=None, load_it=None, n_it=1000, walkers=None):
    cfg['pretrain'] = False
    
    if load_it is not None:
        cfg['load_it'] = load_it

    if run_dir is not None:
        models_path = oj(run_dir, 'models')
        load_file = f'i{load_it}.pk'
        params_path = oj(models_path, load_file)
    
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers=walkers, params_path=params_path)
    walkers, step_size = equilibrate(params, walkers, keys, mol=mol, vwf=vwf, sampler=sampler, compute_energy=False, n_it=100000, step_size_out=True)
    energy_function = create_energy_fn(mol, vwf, separate=True)
    
    if bool(os.environ.get('DISTRIBUTE')) is True:
        energy_function = pmap(energy_function, in_axes=(None, 0))
    
    # step_size = get_step_size(walkers, params, sampler, keys)
    values = {}
    for i in range(n_it):
        keys, subkeys = key_gen(keys)

        walkers, acceptance, step_size = sampler(params, walkers, subkeys, step_size)

        pe, ke = energy_function(params, walkers)
        update_dict(values, 'pe', jnp.mean(pe))
        update_dict(values, 'ke', jnp.mean(ke))
        update_dict(values, 'energy', jnp.mean(pe+ke))
        update_dict(values, 'energy_std', jnp.std(pe+ke))

    n_particles = mol.n_atoms if mol.n_atoms != 0 else mol.n_el
    n_samples = len((values['pe']))
    
    save_values = {}
    save_values['pe_mean_i%i' % load_it] = np.mean(values['pe'])
    save_values['pe_std_i%i' % load_it] = np.std(values['pe'])
    save_values['pe_sem_i%i' % load_it] = np.std(values['pe']) / np.sqrt(n_samples)

    save_values['ke_mean_i%i' % load_it] = np.mean(values['ke'])
    save_values['ke_std_i%i' % load_it] = np.std(values['ke'])
    save_values['ke_sem_i%i' % load_it] = np.std(values['ke']) / np.sqrt(n_samples)

    save_values['e_mean_i%i' % load_it] = np.mean(values['energy'])
    save_values['e_std_i%i' % load_it] = np.std(values['energy'])
    save_values['e_sem_i%i' % load_it] = np.std(values['energy']) / np.sqrt(n_samples)

    save_values['e_std_mean_i%i' % load_it] = np.mean(values['energy_std'])
    save_values['e_std_std_i%i' % load_it] = np.std(values['energy_std'])
    save_values['e_std_sem_i%i' % load_it] = np.std(values['energy_std']) / np.sqrt(n_samples)

    cfg.update(save_values)

    save_config_csv_and_pickle(cfg)
    return cfg


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

