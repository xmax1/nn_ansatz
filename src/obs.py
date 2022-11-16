
import sys
from time import time
from typing import Callable
from itertools import product
from math import ceil, log
import numpy as np

from jax import jit, vmap
import jax.numpy as jnp
import jax.random as rnd

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

from pathlib import Path
from utils import load_pk
from nn_ansatz.systems import SystemAnsatz
from nn_ansatz.ansatz import create_wf
from nn_ansatz.sampling import create_sampler
from functools import partial
from walle_utils import StatsCollector, save_pk
from nn_ansatz.vmc import create_energy_fn


def transform_vector_space(vectors: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis


def split_spin_walkers(walkers, n_up):
    ''' SPLIT BY SPIN 
    walkers: array (n_walkers, n_el, 3)
    n_up: int
    '''
    n_walkers, n_el, n_dim = walkers.shape
    walkers_up = walkers[:, :n_up]
    if n_up == n_el:
        walkers_down = None
    else:
        walkers_down = walkers[:, n_up:]
    return walkers_up, walkers_down




def sample_sphere(
    key, 
    sphere_r,
    n_walkers
):
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/

    k0, k1 = rnd.split(key)
    theta = 2 * jnp.pi * rnd.uniform(k0, (n_walkers, ))
    phi = jnp.arccos(1. - (2. * rnd.uniform(k1, (n_walkers, ))))

    x = jnp.sin( phi) * jnp.cos( theta )
    y = jnp.sin( phi) * jnp.sin( theta )
    z = jnp.cos( phi )

    sphere_sample = jnp.stack([x, y, z], axis=-1)[:, None, :] * sphere_r
    r = jnp.linalg.norm(sphere_sample, axis=-1)

    # if check:
    #     print(f'x {jnp.mean(x)} {jnp.std(x)} \n \
    #             y {jnp.mean(y)} {jnp.std(y)} \n \
    #             z {jnp.mean(z)} {jnp.std(z)} \n \
    #             r {jnp.mean(r)} {jnp.std(r)} {sphere_r} {r.shape} {sphere_sample.shape}')

    return sphere_sample


sample_sphere_1d = lambda key, r, n_walkers: jnp.array([r, 0.0, 0.0])[None, None, :]


def keep_in_boundary(walkers, basis, inv_basis):
    talkers = transform_vector_space(walkers, inv_basis)
    talkers = jnp.fmod(talkers, 1.)
    talkers = jnp.where(talkers < 0., talkers + 1., talkers)
    talkers = transform_vector_space(talkers, basis)
    return talkers


def compute_pr_batch(
    key,
    wb,
    shift,
    params,
    swf, 
    basis,
    inv_basis,
    e_idx,
    n_average_sphere=16
):
    ''' One body density matrix 
    for each walker
    compute over all shifts
    '''
    n_walkers, n_el, _ = wb.shape
    mask = jnp.array([False if i != e_idx else True for i in range(n_el)])[None, :, None]
    
    pr_all = []
    for i in range(n_average_sphere):
        key, subkey = rnd.split(key)
        shifts = sample_sphere(subkey, shift, wb.shape[0])
        walkers_shifted_tmp = jnp.where(mask, wb+shifts, wb)
        walkers_shifted = keep_in_boundary(walkers_shifted_tmp, basis, inv_basis)
        log_psi, sgn = swf(params, wb)
        log_psi_shifted, sgn_shifted = swf(params, walkers_shifted)
        psi = sgn * jnp.exp(log_psi)
        psi_shifted = sgn_shifted * jnp.exp(log_psi_shifted)
        pr_all += [psi_shifted / psi]                                    
    return jnp.mean(jnp.stack(pr_all))


def compute_pr(
    key,
    walkers_batched: list, 
    shift: jnp.ndarray,
    fn: Callable
):
    t0 = time()
    pr_all = []
    for step, w in enumerate(walkers_batched, 1):
        key, subkey = rnd.split(key, 2)
        pr_batch = fn(subkey, w, shift)
        pr_all += [jnp.mean(pr_batch)]
        # t0 = track_time(step, len(walkers_batched), t0)
    return jnp.mean(jnp.array(pr_all))


def cpr_one_body(
    key,
    walker_batched, 
    shift_all, 
    e_idx, 
    basis, 
    inv_basis,
):

    def cpr_batch(
        key_b, 
        wb,
        mask, 
        shift
    ):
        n_walker, n_el, _ = wb.shape

        shift_sphere = sample_sphere(key_b, shift, n_walker)

        wb_shift = jnp.where(mask, wb+shift_sphere, wb)
        wb_shift = keep_in_boundary(wb_shift, basis, inv_basis)
        
        log_psi, sgn = swf(params, wb)
        psi = sgn * jnp.exp(log_psi)

        log_psi_shifted, sgn_shifted = swf(params, wb_shift)
        psi_shifted = sgn_shifted * jnp.exp(log_psi_shifted)
        
        return jnp.mean(psi_shifted / psi)

    jit_cpr_batch = jit(cpr_batch)

    pr_all = []
    for r in shift_all:
        pr = 0.0
        for sub_e_idx in range(e_idx, e_idx+7):
            mask = jnp.array([False if i != sub_e_idx else True for i in range(n_el)])[None, :, None]
            for step, w in enumerate(walker_batched, 1):
                w = jnp.array(w)
                key, subkey = rnd.split(key)
                pr += jit_cpr_batch(subkey, w, mask, r)
        pr /= (float(step)*7.)
        pr_all += [pr]
    return np.array(pr_all)


def cartesian_product(*args) -> list:
    ''' Cartesian product is the ordered set of all combinations of n sets '''
    return list(product(*args))


def make_arg_key(key):
    key = key.replace('-', '')
    return key


def collect_args():
    if len(sys.argv) == 1:
        args = {}
    else:
        args = ' '.join(sys.argv[1:])
        args = args.split('--')[1:]  # first element is blank
        args = [a.split(' ', 1) for a in args]
        args = iter([x.replace(' ', '') for sub_list in args for x in sub_list])
        args = {make_arg_key(k):v for k, v in zip(args, args)}
    return args
    

def track_time(
    step: int, 
    total_step: int, 
    t0: float, 
    every_n: int=10,
    tag: str=''
):
    """ Prints time
    
    """
    if step % every_n == 0:
        d, t0 = time() - t0, time()
        print(
            f'{tag} | \
            step {step} % complete: {step/float(total_step):.2f} | \
            t: {(d):.2f} |'
        )
    return t0  # replaces t0 external


def equilibrate_1M_walkers(
    key, 
    n_walkers_max, 
    walkers, 
    params,
    sampler, 
    compute_energy,
    n_warm_up_steps=1000,
    n_walkers_eq_max=256, 
    e_every=25, 
    step_size=0.05,
    n_eq_walkers=200000,
):
    print('WARMING UP EQ')
    t0 = time()
    for step in range(n_warm_up_steps):
        key, subkey = rnd.split(key, 2)
        walkers, acc, step_size = sampler(params, walkers, subkey, step_size)
        
        if step % e_every == 0:
            pe, ke = compute_energy(params, walkers)
            print(f'Step: {step} | E: {jnp.mean(pe+ke):.7f} +- {jnp.std(pe+ke):.7f}')
            print(f'Step_size: {float(step_size):.3f} | Acc.: {float(acc):.2f}')
        
        t0 = track_time(step, n_warm_up_steps, t0, every_n=e_every)

    print('EQUILIBRATION LOOP')
    n_batch = int((n_eq_walkers//n_walkers_max))
    ex_stats = StatsCollector()
    w = walkers
    for step in range(1, n_batch+1):
    
        for i in range(10):
            key, subkey = rnd.split(key, 2)
            w, acc, step_size = sampler(params, w, subkey, step_size)
        
        walkers = jnp.concatenate([walkers, w], axis=0)
        
        if step % e_every == 0:
            pe, ke = compute_energy(params, w)
            ex_stats.ke = jnp.mean(ke)
            ex_stats.pe = jnp.mean(pe)
            ex_stats.e = jnp.mean(ke + pe)
            print(f'Step: {step} | E: {jnp.mean(ke + pe):.6f} | n_eq: {len(walkers)}')
            print(f'Step_size: {float(step_size):.3f} | Acc.: {float(acc):.2f}')
    try:
        print(np.mean(ex_stats.e), 'e', np.std(ex_stats.e))
    except:
        pass

    return walkers


closest = lambda x: int(ceil(log(x) / log(2)))

def input_bool(x):
    from distutils.util import strtobool
    x = strtobool(x)
    if x: return True
    else: return False

args = collect_args()

# print(seed := int(args['seed']))
print(run_dir := Path(args['run_dir']))
print(n_walkers_max := int(args['n_walkers_max']))
print(equilibrate := input_bool(args['equilibrate']))
print(n_walkers := int(args['n_walkers']))
print(n_average_sphere := int(args['n_average_sphere']))
print(n_points := int(args['n_points']))
print(n_dim_exp := int(args['n_dim_exp']))
print(n_points_hypercube := int(args['n_points_hypercube']))
print(load_it := int(args['load_it']))
print(walkers_name := str((args["walkers_name"])))
print(compute_energy_from_walkers := str((args["compute_energy_from_walkers"])))
n_k_basis = 2

cfg = load_pk(run_dir / 'config1.pk')
n_el, n_up, seed = cfg['n_el'], cfg['n_up'], cfg['seed']

key = rnd.PRNGKey(seed)

# results_dir = Path('/scratch/amawi', run_dir)
res_dir = run_dir / 'res_final'
if not res_dir.exists():
    res_dir.mkdir()
results_path = res_dir / f'obs_final_d{n_dim_exp}_{n_walkers//1000}k_{n_average_sphere}av_i{load_it}.pk'
models_path = run_dir / 'models'

mol = SystemAnsatz(**cfg)
vwf = jit(create_wf(mol))
swf = jit(create_wf(mol, signed=True))
sampler = create_sampler(mol, vwf)
params = load_pk(models_path / f'i{load_it}.pk')
compute_energy = jit(create_energy_fn(mol, vwf, separate=True))

step_size = 0.04
if ((w_path := run_dir/walkers_name)).exists():
    try:
        walkers = np.array(load_pk(w_path)['walkers'])
    except:
        walkers = np.array(load_pk(w_path))
    if walkers.ndim == 4:
        walkers = np.squeeze(walkers.reshape(-1, *walkers.shape[-2:]))
else:
    walkers = load_pk(run_dir / 'models' / f'w_tr_i{load_it}.pk')
    if walkers.ndim == 4:
        walkers = np.squeeze(walkers[0])
    w = walkers = jnp.array(walkers[:n_walkers_max])
    n_eq = int(n_walkers//n_walkers_max) + 1
    for step in range(1, n_eq):
        for _ in range(10):
            key, subkey = rnd.split(key)
            w, acc, step_size = sampler(params, w, subkey, step_size)
        walkers = jnp.concatenate([walkers, w])
    walkers = np.array(walkers)
    save_pk(walkers, run_dir / f'eq_w_{n_walkers//1000}k_i{load_it}.pk')

# expstats save not ex stats
print('LOADING WALKERS: ', w_path, 'SHAPE: ', walkers.shape)
walkers = walkers[:min(len(walkers), n_walkers)]
n_walkers = len(walkers)
n_batch = (n_walkers//n_walkers_max) - 1
n_walkers = (n_batch*n_walkers_max)
walkers = walkers[:n_walkers]
print('USING WALKERS SHAPE: ', walkers.shape, 'N_BATCH: ', n_batch)
walkers_batched = np.split(walkers, n_batch, axis=0)

exp_stats = StatsCollector()
print('COMPUTING ENERGY')
if compute_energy_from_walkers:
    for step, w in enumerate(walkers_batched, 1):
        w_gpu = jnp.array(w)
        pe, ke = compute_energy(params, w_gpu)
        exp_stats.ke = np.array(jnp.squeeze(ke))
        exp_stats.pe = np.array(jnp.squeeze(pe))
        exp_stats.e = np.array(jnp.squeeze(ke + pe))
        del w_gpu
        try:
            if step % 100 == 0:
                print(f'Step: {step} | Energy: {np.mean(exp_stats.get("e")):.7f} +- {np.std(exp_stats.get("e")):.7f}') 
        except Exception as err:
            print(err)

save_pk(exp_stats.get_dict(), results_path)

max_distance = mol.scale_cell / 2.
rs = jnp.linspace(0., max_distance, n_points)
print('Max distance: ', max_distance, 'Scale: ', mol.scale_cell, 'n_walkers: ', n_walkers)

exp_stats.rs_one_body = rs
print('COMPUTING ONE BODY DENSITY MATRIX')
e_idxs_one_body = [0 if n_up == n_el else 0, 7]
for e_idx in e_idxs_one_body:
    x_name = f'pr_x_{e_idx}'
    y_name = f'pr_{e_idx}'
    pr = cpr_one_body(key, walkers_batched, rs, e_idx, mol.basis, mol.inv_basis)
    exp_stats.set_by_name(y_name, pr)

save_pk(exp_stats.get_dict(), results_path)

print('COMPUTING MOMENTUM DISTRIBUTION')

'''
saverio pseudocode for n(k)

initialize a list of k vectors
initialize a list of configurations
initialize to zero the momentum distribution n(k)
initialize to zero a counter c(k)
for each configuration:
    calculate psi
    for each particle:
        for i in range(1, whatever): [you can displace this particle more than once]
            sample deltar uniformly in a cube of side L centered at this particle
            displace this particle by deltar
            calculate psi'
            for each k vector:
                add 1 to c(k)
                add psi'/psi * cos(k\dot deltar) to n(k)
            restore position of this particle
for each k vector:
    n(k)=n(k)/c(k)
    print k,n(k)
'''
n_x_reciprocal = tuple(range(-n_k_basis, n_k_basis+1))
n_y_reciprocal = n_x_reciprocal if n_dim_exp == 3 else (0,)
n_z_reciprocal = n_x_reciprocal if n_dim_exp == 3 else (0,)
kpoints_int_all = np.array(cartesian_product(n_x_reciprocal, n_y_reciprocal, n_z_reciprocal))

## Filters out same norm
# un_kpoints = []
# for k in kpoints_int_all:
#     k_sort = np.sort(np.absolute(k))
#     not_in_there = any([np.all(k_sort == np.sort(np.absolute(k_tmp))) for k_tmp in un_kpoints])
#     if not_in_there:
#         continue
#     un_kpoints += [k]
# kpoints_int = np.array(un_kpoints)
kpoints_int = np.copy(kpoints_int_all)
kpoints = (jnp.array(kpoints_int) * 2 * np.pi) / mol.volume**(1/3.)
kpoints_names = np.array(['_'.join([str(i) for i in kpoint]) for kpoint in kpoints_int])

print(kpoints)
exp_stats.kpoints = kpoints
exp_stats.kpoints_int = kpoints_int
exp_stats.kpoints_names = kpoints_names

# x_hyperspace = jnp.linspace(0., max_distance, n_points_hypercube)[1:-1]
# y_hyperspace = x_hyperspace if n_dim_exp == 3 else jnp.array([0.])
# z_hyperspace = x_hyperspace if n_dim_exp == 3 else jnp.array([0.])
# rs_hyperspace = jnp.array(cartesian_product(x_hyperspace, y_hyperspace, z_hyperspace))

e_idxs = [0, 7]
n_disp = 10
for e_idx in e_idxs:
    x_name = f'pr_r_mom_{e_idx}'
    y_name = f'pr_mom_{e_idx}'
    y_nk_name = f'nk_{e_idx}'

    nk = []
    for step, w in enumerate(walkers_batched, 1):
        w = jnp.array(w)
        n_w, n_el, _ = w.shape
        center = w/2.

        for sub_e_idx in range(e_idx, e_idx+7):
            mask = jnp.array([False if i != sub_e_idx else True for i in range(n_el)])[None, :, None]
            
            for sub_step in range(n_disp):
                key, subkey = rnd.split(key)
                shift_cube = rnd.uniform(subkey, shape=(n_w, 1, 3), minval=0.0, maxval=mol.scale_cell) - center
                wb_first = jnp.where(mask, w+shift_cube, w)
                wb_shift = keep_in_boundary(wb_first, mol.basis, mol.inv_basis)
                
                log_psi, sgn = swf(params, w)
                psi = sgn * jnp.exp(log_psi)

                log_psi_shifted, sgn_shifted = swf(params, wb_shift)
                psi_shifted = sgn_shifted * jnp.exp(log_psi_shifted)
                
                pr = (psi_shifted / psi).reshape(n_w, 1)

                nk += [jnp.mean(pr * (jnp.cos(jnp.squeeze(shift_cube[:, e_idx, :]) @ jnp.transpose(kpoints))), axis=0, keepdims=True)] # (nk, 3) @ (3, nw)
        
    nk = jnp.mean(jnp.array(nk), axis=0)  # / mol.volume
    exp_stats.set_by_name(y_nk_name, nk)

save_pk(exp_stats.get_dict(), results_path)

print('COMPUTING PAIR CORRELATION')
def drop_diagonal_i(square):
    n = square.shape[0]
    split1 = jnp.split(square, n, axis=0)
    upper = [jnp.split(split1[i], [j], axis=1)[1] for i, j in zip(range(0, n), range(1, n))]
    lower = [jnp.split(split1[i], [j], axis=1)[0] for i, j in zip(range(1, n), range(1, n))]
    arr = [ls[i] for i in range(n-1) for ls in (upper, lower)]
    result = jnp.concatenate(arr, axis=1)
    return jnp.squeeze(result).reshape((n, n-1))

drop_diagonal = vmap(drop_diagonal_i, in_axes=(0,))

def transform_vector_space(vectors: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    if basis.shape == (3, 3):
        return jnp.dot(vectors, basis)
    else:
        return vectors * basis

def apply_minimum_image_convention(displacement_vectors, basis, inv_basis):
    displace = (2. * transform_vector_space(displacement_vectors, inv_basis)).astype(int).astype(displacement_vectors.dtype)
    displace = transform_vector_space(displace, basis)
    displacement_vectors = displacement_vectors - displace
    return displacement_vectors


def min_im(disp, basis, inv_basis):
    trs_disp = jnp.dot(disp, inv_basis)
    trs_min_im_disp = 2.*trs_disp  # absolute floor operation
    sgn_disp = jnp.sign(trs_min_im_disp)
    shift = sgn_disp * jnp.floor(jnp.absolute(trs_min_im_disp))
    trs_shift = jnp.dot(shift, basis)
    disp = disp - trs_shift
    return disp
    
@jit
def compute_distances(w_0, w_1, basis, inv_basis):
    disp = jnp.expand_dims(w_0, axis=-2) - jnp.expand_dims(w_1, axis=-3)
    disp = min_im(disp, basis, inv_basis)
    return jnp.linalg.norm(disp, axis=-1)  # (n_walkers, n_i, n_j)

@jit
def compute_gr_hist(rs, distances, volume):
    '''
    INFO: http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html
    '''
    n_walker, n_ri, n_rj = distances.shape # number of reference particles, number target
    number_density = n_rj / volume # the number of electrons  
    volume_element = (4.*jnp.pi/3.) * (rs[1:]**3 - rs[:-1]**3)
    n_bins = len(volume_element)
    hist, bins = jnp.histogramdd(distances.reshape(-1, 1), bins=n_bins, range=[(rs[0], rs[-1]),])
    pdf = hist / (n_walker * volume_element * number_density * n_ri)
    return pdf

@jit
def compute_gr(rs, distances, volume):
    '''
    INFO: http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html
    '''
    n_w, n_ri, n_rj = distances.shape # number of reference particles, number target
    number_density = n_rj / volume # the number of electrons  
    
    vol_els = (4.*jnp.pi/3.) * (rs[1:]**3 - rs[:-1]**3)

    pdf_all = []
    for r0, r1, v in zip(rs[:-1], rs[1:], vol_els):
        gr_th_r0 = distances > r0
        le_th_r1 = distances < r1
        btw = gr_th_r0 * le_th_r1
        n_count = jnp.sum(btw)
        pdf_all += [n_count / (n_w * v * number_density * n_ri)]

    return jnp.array(pdf_all)

rs = jnp.linspace(0., max_distance, n_points)[1:]
exp_stats.rs_gr = rs
for step, w in enumerate(walkers_batched, 1):
    w = jnp.array(w)
    w_up, w_down = jnp.split(w, 2, axis=1)

    d_up = compute_distances(w_up, w_up, mol.basis, mol.inv_basis)
    d_down = compute_distances(w_up, w_down, mol.basis, mol.inv_basis)
    d_up_down = compute_distances(w_down, w_down, mol.basis, mol.inv_basis)

    exp_stats.gr_up = compute_gr(rs, d_up, mol.volume)[None, :]
    exp_stats.gr_down = compute_gr(rs, d_down, mol.volume)[None, :]
    exp_stats.gr_up_down = compute_gr(rs, d_up_down, mol.volume)[None, :]

    exp_stats.gr_up_hist = compute_gr_hist(rs, d_up, mol.volume)[None, :]
    exp_stats.gr_down_hist = compute_gr_hist(rs, d_down, mol.volume)[None, :]
    exp_stats.gr_up_down_hist = compute_gr_hist(rs, d_up_down, mol.volume)[None, :]

exp_stats.process(['gr_up', 'gr_down', 'gr_up_down'], apply_fn=lambda x: np.mean(x, axis=0))
exp_stats.process(['gr_up_hist', 'gr_down_hist', 'gr_up_down_hist'], apply_fn=lambda x: np.mean(x, axis=0))

print('SAVING THINGS')
save_pk(exp_stats.get_dict(), results_path)

print(exp_stats.get('gr_up'))
print(exp_stats.get('gr_up_hist'))

# ''' PROOFS 
# import numpy as np
# from itertools import product

# def compute_reciprocal_basis(basis, volume):
#     cv1, cv2, cv3 = np.split(basis, 3, axis=0)
#     rv1 = np.cross(cv2.squeeze(), cv3.squeeze()) / volume
#     rv2 = np.cross(cv3.squeeze(), cv1.squeeze()) / volume
#     rv3 = np.cross(cv1.squeeze(), cv2.squeeze()) / volume
#     reciprocal_basis = np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)
#     return reciprocal_basis * 2 * np.pi

# a, b, c = 3.5, 3.5, 3.5
# m = 2
# basis = np.diag([a, b, c])
# volume = a*b*c
# length = volume**(1/3.)
# kdomain = np.arange(-m, m+1)
# comb = np.array(list(product(*[kdomain for _ in range(3)])))
# reciprocal_basis = compute_reciprocal_basis(basis, volume)  # 3 x 3
# kpoints = comb @ reciprocal_basis

# # k = 2π(m1, m2, m3)/L
# kpoints_from_ints = comb * 2 * np.pi / length

# for i, j in zip(kpoints, kpoints_from_ints):
#     print(i-j)
# ''' 