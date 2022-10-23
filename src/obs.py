
import sys
from time import time
from typing import Callable
from itertools import product
from math import ceil, log


import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as rnd

from pathlib import Path
from utils import load_pk
from nn_ansatz.systems import SystemAnsatz
from nn_ansatz.ansatz import create_wf
from nn_ansatz.sampling import create_sampler, generate_walkers
from functools import partial
from walle_utils import StatsCollector, save_pk
from nn_ansatz.vmc import create_energy_fn


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


def compute_distances(walkers: jnp.ndarray, mol, rj=None):
    if walkers is None:
        return jnp.array([])

    if rj is None:
        rj_walkers = walkers
        dd = drop_diagonal
    else:
        rj_walkers = rj
        dd = lambda x: x

    displacement = walkers[..., None, :] - jnp.expand_dims(rj_walkers, axis=-3)
    displacement = apply_minimum_image_convention(displacement, mol.basis, mol.inv_basis)
    distances = dd(jnp.linalg.norm(displacement, axis=-1))
    return distances  # (n_walkers, n_i, n_j)


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


def compute_gr(rs, distances, volume):
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


def sample_sphere(
    key, 
    sphere_r,
    n_walkers
):
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/

    subkeys = rnd.split(key, 3)

    theta = 2 * jnp.pi * rnd.uniform(subkeys[0], (n_walkers, ))
    phi = jnp.arccos(1. - (2. * rnd.uniform(subkeys[1], (n_walkers, ))))

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


def robust_wrapper(fn):
    """ Wrapper for 'non-core' code - it does not matter if it breaks
    Returns nones for days
    """
    def new_fn(*args, n_out=1, msg='Error in a non-core function'):
        try:
            out = fn(*args)
        except Exception as e:
            print(msg, e)   
            out = [None for _ in range(n_out)]
        return out
    return new_fn
    

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
            % complete: {step/float(total_step):.2f} | \
            t: {(d):.2f} |'
        )
    return t0  # replaces t0 external

closest = lambda x: int(ceil(log(x) / log(2)))

def input_bool(x):
    from distutils.util import strtobool
    x = strtobool(x)
    if x: return True
    else: return False

args = collect_args()

seed = int(args['seed'])
run_dir = Path(args['run_dir'])
n_walkers_max = int(args['n_walkers_max'])
equilibrate = input_bool(args['equilibrate'])
n_walkers = int(args['n_walkers'])
n_average_sphere = int(args['n_average_sphere'])
n_points = int(args['n_points'])
n_dim_exp = int(args['n_dim_exp'])
n_points_hypercube = int(args['n_points_hypercube'])
n_k_basis = 2

key = rnd.PRNGKey(seed)

results_dir = Path('/scratch/amawi', run_dir)
results_path = results_dir / f'obs_d{n_dim_exp}_{n_walkers//1000}k_{n_average_sphere}av.pk'

cfg = load_pk(run_dir / 'config1.pk')
n_el, n_up = cfg['n_el'], cfg['n_up']
energy_all = {k:v for k,v in cfg.items() if 'equilibrated_energy_mean_i' in k}
params_key = min(energy_all, key=cfg.get)    
load_it = int(params_key.split('i')[-1])

print(f'rs = {cfg["density_parameter"]}, step={params_key.split("i")[-1]}, final_e = {2*cfg[params_key]:.5f}')

models_path = run_dir / 'models'
walkers_dir = run_dir / 'walkers'
load_file = f'i{load_it}.pk'

mol = SystemAnsatz(**cfg)
vwf = jit(create_wf(mol))
swf = jit(create_wf(mol, signed=True))
sampler = create_sampler(mol, vwf)
params = load_pk(models_path / f'i{load_it}.pk')

compute_pr_e0 = jit(partial(compute_pr_batch, params=params, swf=swf, basis=mol.basis, inv_basis=mol.inv_basis, 
                                                    e_idx=0, n_average_sphere=n_average_sphere))
compute_pr_e7 = jit(partial(compute_pr_batch, params=params, swf=swf, basis=mol.basis, inv_basis=mol.inv_basis, 
                                                    e_idx=7, n_average_sphere=n_average_sphere))

compute_prs = {
    0: partial(compute_pr, fn=compute_pr_e0),
    7: partial(compute_pr, fn=compute_pr_e7)
}

_compute_distances = jit(partial(compute_distances, mol=mol))
_compute_gr = jit(partial(compute_gr, volume=mol.volume))
compute_energy = jit(create_energy_fn(mol, vwf, separate=True))


def equilibrate_1M_walkers(
    key, 
    n_walkers_max, 
    walkers, 
    n_warm_up_steps=100,
    n_walkers_eq_max=256, 
    e_every=5, 
    step_size=0.5,
    n_eq_walkers=1000000,
):
    
    print('WARMING UP EQ')
    t0 = time()
    for step in range(n_warm_up_steps):
        key, subkey = rnd.split(key, 2)
        walkers, acc, step_size = sampler(params, walkers, subkey, step_size)
        t0 = track_time(step, n_warm_up_steps, t0, every_n=50)

    print('EQUILIBRATION LOOP')
    n_batch = int((n_eq_walkers//n_walkers_max))

    ex_stats = StatsCollector()
    for step in range(1, n_batch+1):
        key, subkey = rnd.split(key, 2)

        idxs_all = jnp.arange(0, len(walkers))
        idxs = rnd.choice(subkey, idxs_all, (n_walkers_max, ))
        w = walkers[idxs]

        w, acc, step_size = sampler(params, w, subkey, step_size)
        
        walkers = jnp.concatenate([walkers, w], axis=0)
        
        if step % e_every == 0:
            walkers_tmp = w[:n_walkers_eq_max]
            pe, ke = compute_energy(params, walkers_tmp)
            ex_stats.ke = jnp.mean(ke)
            ex_stats.pe = jnp.mean(pe)
            ex_stats.e = jnp.mean(ke + pe)
            print(f'Step: {step} | E: {jnp.mean(ke + pe):.6f} | n_eq: {len(walkers)}')

    save_pk(ex_stats.get_dict(), results_dir / f'e_stats_i{load_it}.pk')

    return walkers

equilibrated_walkers_path = Path(run_dir, f'eq_walkers_i{load_it}.pk')

max_distance = mol.scale_cell / 2.
rs = jnp.linspace(0., max_distance, n_points)
print('Max distance: ', max_distance, 'Scale: ', mol.scale_cell, 'n_walkers: ', n_walkers)

if equilibrate:
    print('EQUILIBRATING WALKERS')
    walkers = rnd.uniform(key, (n_walkers_max, n_el, 3), minval=0., maxval=max_distance*2)
    # walkers = jnp.array(load_pk('/home/energy/amawi/projects/nn_ansatz/src/experiments/PRX_Responses/runs/run41035/eq_walkers_i100000.pk'))
    _ = sampler(params, walkers, key, 0.02)
    walkers = equilibrate_1M_walkers(key, n_walkers_max, walkers[:n_walkers_max])
    save_pk(np.array(walkers), equilibrated_walkers_path)
else:
    print('LOADING WALKERS') 
    walkers = jnp.array(load_pk(equilibrated_walkers_path))

walkers = walkers[:min(n_walkers, len(walkers))]
print(n_walkers, n_el, _ := walkers.shape)

n_batch = n_walkers//n_walkers_max
walkers = walkers[:(n_batch*n_walkers_max)]
walkers_batched = jnp.split(walkers, n_batch, axis=0)

exp_stats = StatsCollector()
exp_stats.rs_one_body = rs

print('COMPUTING ONE BODY DENSITY MATRIX')
e_idxs_one_body = [0 if n_up == n_el else 0, 7]

for e_idx in e_idxs_one_body:
    x_name = f'pr_x_{e_idx}'
    y_name = f'pr_{e_idx}'
    _compute_pr = compute_prs[e_idx]
    
    t0 = time()
    for step, r in enumerate(rs, 1):
        key, subkey = rnd.split(key, 2)
        pr = _compute_pr(subkey, walkers_batched, r)
        exp_stats.set_by_name(y_name, pr)
        t0 = track_time(step, len(rs), t0)


print('COMPUTING MOMENTUM DISTRIBUTION')
n_x_reciprocal = tuple(range(-n_k_basis, n_k_basis+1))
n_y_reciprocal = n_x_reciprocal if n_dim_exp == 3 else (0,)
n_z_reciprocal = n_x_reciprocal if n_dim_exp == 3 else (0,)
kpoints_int_all = jnp.array(cartesian_product(n_x_reciprocal, n_y_reciprocal, n_z_reciprocal))

### Filters out same norm
# kpoints_int = []
# for k in kpoints_int_all:
#     if any([set(k) == set(k_tmp) for k_tmp in kpoints_int]):
#         continue
#     kpoints_int += [k]
kpoints_int = kpoints_int_all

kpoints = jnp.array(kpoints_int) * 2 * np.pi / mol.volume**(1/3.)
kpoints_names = np.array(['_'.join([str(i) for i in kpoint]) for kpoint in kpoints_int])

exp_stats.kpoints = kpoints
exp_stats.kpoints_int = kpoints_int
exp_stats.kpoints_names = kpoints_names

x_hyperspace = jnp.linspace(0., max_distance, n_points_hypercube)[1:-1]
y_hyperspace = x_hyperspace if n_dim_exp == 3 else jnp.array([0.])
z_hyperspace = x_hyperspace if n_dim_exp == 3 else jnp.array([0.])
rs_hyperspace = jnp.array(cartesian_product(x_hyperspace, y_hyperspace, z_hyperspace))

e_idxs = [0, 7]
for e_idx in e_idxs:
    x_name = f'pr_r_mom_{e_idx}'
    y_name = f'pr_mom_{e_idx}'
    y_nk_name = f'nk_{e_idx}'

    _compute_pr = compute_prs[e_idx]

    for step, r_vec in enumerate(rs_hyperspace, 1):
        key, subkey = rnd.split(key, 2)
        r = jnp.linalg.norm(r_vec, -1)
        pr = _compute_pr(subkey, walkers_batched, r)
        w = jnp.exp(1j * (kpoints @ r_vec))
        
        exp_stats.set_by_name(y_nk_name, (w * pr)[None, :])
        exp_stats.set_by_name(x_name, r_vec[None, :])
        exp_stats.set_by_name(y_name, pr)

        t0 = track_time(step, len(rs_hyperspace), t0)

    exp_stats.process(y_nk_name, apply_fn=lambda x: np.mean(x, axis=0) / mol.volume)

        
print('COMPUTING PAIR CORRELATION')
rs_pair = rs[1:]
for step, w in enumerate(walkers_batched, 1):
    w_up, w_down = split_spin_walkers(w, n_up)

    d_up = _compute_distances(w_up)
    d_down = _compute_distances(w_down)
    d_up_down = _compute_distances(w_up, rj=w_down)

    exp_stats.gr_up = _compute_gr(rs_pair, d_up)[None, :]
    exp_stats.gr_down = _compute_gr(rs_pair, d_down)[None, :]
    exp_stats.gr_up_down = _compute_gr(rs_pair, d_up_down)[None, :]

    t0 = track_time(step, len(walkers_batched), t0)

exp_stats.process(['gr_up', 'gr_down', 'gr_up_down'], apply_fn=lambda x: np.mean(x, axis=0))

print('SAVING THINGS')

save_pk(exp_stats.get_dict(), results_path)

''' PROOFS 
import numpy as np
from itertools import product

def compute_reciprocal_basis(basis, volume):
    cv1, cv2, cv3 = np.split(basis, 3, axis=0)
    rv1 = np.cross(cv2.squeeze(), cv3.squeeze()) / volume
    rv2 = np.cross(cv3.squeeze(), cv1.squeeze()) / volume
    rv3 = np.cross(cv1.squeeze(), cv2.squeeze()) / volume
    reciprocal_basis = np.concatenate([x[None, :] for x in (rv1, rv2, rv3)], axis=0)
    return reciprocal_basis * 2 * np.pi

a, b, c = 3.5, 3.5, 3.5
m = 2
basis = np.diag([a, b, c])
volume = a*b*c
length = volume**(1/3.)
kdomain = np.arange(-m, m+1)
comb = np.array(list(product(*[kdomain for _ in range(3)])))
reciprocal_basis = compute_reciprocal_basis(basis, volume)  # 3 x 3
kpoints = comb @ reciprocal_basis

# k = 2π(m1, m2, m3)/L
kpoints_from_ints = comb * 2 * np.pi / length

for i, j in zip(kpoints, kpoints_from_ints):
    print(i-j)
''' 