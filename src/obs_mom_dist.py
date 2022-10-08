
from nn_ansatz.utils import save_config_csv_and_pickle, save_pk
from nn_ansatz.sampling import equilibrate, keep_in_boundary
from utils import load_pk, oj, ojm
from nn_ansatz.ansatz import create_wf
from nn_ansatz.routines import initialise_system_wf_and_sampler
from nn_ansatz.vmc import create_energy_fn
from nn_ansatz.systems import SystemAnsatz, compute_distance, generate_k_points
from nn_ansatz.plot import plot, format_ax, format_fig
from nn_ansatz.ansatz_base import apply_minimum_image_convention
from walle_utils import StatsCollector, ojm, format_ax, format_fig

from jax import random as rnd, numpy as jnp
import numpy as np
from jax import grad, jit
from functools import partial
from itertools import product
from typing import Union, Callable, Iterable
import os
from matplotlib import pyplot as plt
import time


def compute_distances(walkers: jnp.ndarray, mol, rj=None):
    if walkers is None:
        return jnp.array([])

    if rj is None:
        rj_walkers = walkers
    else:
        rj_walkers = rj

    displacement = walkers[..., None, :] - jnp.expand_dims(rj_walkers, axis=-3)
    displacement = apply_minimum_image_convention(displacement, mol.basis, mol.inv_basis, on=True)
    distances = jnp.linalg.norm(displacement, axis=-1)
    return distances  # (n_walkers, n_i, n_j)


def split_spin_walkers(walkers, n_up):
    walkers_up = walkers[..., :n_up, :]
    if n_up == walkers.shape[-2]:
        walkers_down = None
    else:
        walkers_down = walkers[..., n_up:, :]
    return walkers_up, walkers_down


def compute_gr(rs, distances, volume, n_points):
        '''
        http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html
        '''
        n_walkers = len(distances)
        n_ri, n_rj = distances.shape[1:]  # number of reference particles, number target
        number_density = n_rj / volume  # is the number density n_ri or 
        
        volume_element = (4.*jnp.pi/3.) * (rs[1:]**3 - rs[0:-1]**3)
        hist, bins = jnp.histogramdd(distances.reshape(-1, 1), bins=n_points-1, range=[(0, rs[-1]),])
        pdf = hist / (n_walkers * volume_element * number_density)

        # pdfs = []
        # for i, r in enumerate(rs):
        #     r0 = rs[i]
        #     r1 = rs[i+1]
        #     counts = float(jnp.sum((r0<=distances)*(distances<=r1))) / n_ri
        #     volume_element = (4.*np.pi/3.) * (r1**3 - r0**3)
        #     # volume = 4.* np.pi * r0**2 * dr  # less accurate
        #     pdf = counts / (n_walkers * volume_element * number_density)
        #     pdfs.append(pdf)

        return pdf


def sample_sphere(key, sphere_r, n_walkers=256, check=False):
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/

    subkeys = rnd.split(key, 3)

    theta = 2*jnp.pi * rnd.uniform(subkeys[0], (n_walkers, ))
    phi = jnp.arccos(1. - (2. * rnd.uniform(subkeys[1], (n_walkers, ))))

    x = jnp.sin( phi) * jnp.cos( theta )
    y = jnp.sin( phi) * jnp.sin( theta )
    z = jnp.cos( phi )

    sphere_sample = jnp.stack([x, y, z], axis=-1)[:, None, :] * sphere_r
    r = jnp.linalg.norm(sphere_sample, axis=-1)

    if check:
        print(f'x {jnp.mean(x)} {jnp.std(x)} \n \
                y {jnp.mean(y)} {jnp.std(y)} \n \
                z {jnp.mean(z)} {jnp.std(z)} \n \
                r {jnp.mean(r)} {jnp.std(r)} {sphere_r} {r.shape} {sphere_sample.shape}')

    return sphere_sample


def cut_outliers(x: np.ndarray, y: np.ndarray, frac_of_data=0.95, passthrough=False):
    if passthrough:
        return x, y
    n_data = len(x)
    bottom = int((1.-frac_of_data)*n_data)
    top = int(frac_of_data*n_data)
    idxs = np.argsort(y)[bottom:top]  # low to high
    mean, std = np.mean(y[idxs]), np.std(y[idxs])
    bar = 10*std
    mask = ((mean-bar) < y) * (y < (mean+bar))
    y = y[mask]
    x = x[mask]
    return x, y

    
def compute_pr(walkers: jnp.ndarray, 
               shift: jnp.ndarray, 
               params: dict, 
               swf: Callable, 
               basis: jnp.ndarray, 
               inv_basis: jnp.ndarray,
               e_idx: int,
               n_el: int):
    ''' One body density matrix '''
    mask = jnp.array([False if i != e_idx else True for i in range(n_el)])[None, :, None]
    walkers_r = jnp.where(mask, walkers+shift, walkers)
    walkers_r = keep_in_boundary(walkers, basis, inv_basis)
    log_psi, sgn = swf(params, walkers)
    log_psi_r, sgn_r = swf(params, walkers_r)
    psi = sgn * jnp.exp(log_psi)
    psi_r = sgn_r * jnp.exp(log_psi_r)
    pr = psi_r / psi
    return pr


def generate_nd_hypercube(lims, n_points=20):
    points = [jnp.linspace(*lim, n_points) for lim in lims]
    hypercube = jnp.array(list(product(*points)))
    return hypercube


def get_cartesian_product(*args):
    '''
    Cartesian product is the ordered set of all combinations of n sets
    '''
    return list(product(*args))


def generate_kpoints_by_m(m=2):
    kdomain = jnp.arange(-m, m+1)
    kpoints = jnp.array(get_cartesian_product(kdomain, kdomain, kdomain))
    return kpoints * 2 * jnp.pi


def get_fig_size(n_col, n_row, ratio=0.75, base=5, scaling=0.85):
        additional_space_a = [base * scaling**x for x in range(1, n_col+1)]
        additional_space_b = [ratio * base * scaling**x for x in range(1, n_row+1)]
        return (sum(additional_space_a), sum(additional_space_b))


def compute(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
            load_it: int=100000,
            seed: int=0,
            n_batch: int=10,
            n_points: int=50,
            plot_dir: str=None,
            n_walkers_max: int=256,
            n_walkers: int=None,
            walkers: Union[np.ndarray, None] = None,
            d3: bool = True,
            **exp         
    ):

    key = rnd.PRNGKey(seed)

    models_path = oj(run_dir, 'models')
    walkers_dir = oj(run_dir, 'walkers')
    load_file = f'i{load_it}.pk'

    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']

    walkers_dir = oj(run_dir, 'walkers')
    walkers_path = ojm(walkers_dir, load_file)
    if not os.path.exists(walkers_path):
        walkers_path = None

    params_path = oj(models_path, load_file)
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, params_path=params_path, walkers_path=walkers_path)
    swf = jit(create_wf(mol, signed=True))

    cut = int((100000//n_walkers_max) * n_walkers_max)
    all_walkers = jnp.array(load_pk(ojm(run_dir, f'eq_walkers_i{load_it}.pk')))
    all_walkers = all_walkers[cut:cut+int((n_walkers//n_walkers_max) * n_walkers_max)] if n_walkers is not None else all_walkers[cut:]
    n_walkers_all = len(all_walkers)
    all_walkers = jnp.split(all_walkers, len(all_walkers)//n_walkers_max, axis=0)
    print('n_walkers_all ', n_walkers_all)

    max_distance = mol.scale_cell
    print('Max distance: ', max_distance, 'Scale: ', mol.scale_cell)

    compute_prs = {
        0: jit(partial(compute_pr, swf=swf, e_idx=0, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis)),
        7: jit(partial(compute_pr, swf=swf, e_idx=7, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis))
    }
    n_steps = 10
    _sample_sphere = jit(partial(sample_sphere, n_walkers=n_walkers_max))
    _compute_distances = jit(partial(compute_distances, mol=mol))
    _compute_gr = jit(partial(compute_gr, volume=mol.volume, n_points=n_points))

    exp_stats = StatsCollector()

    ''' COMPUTE MOMENTUM DISTRIBUTION '''
    print('Computing momentum distribution')
    
    if d3:
        rs = generate_nd_hypercube(([0., max_distance], [0., max_distance], [0., max_distance]), n_points=10)
        kpoints_int = jnp.rint(generate_kpoints_by_m(m=2) / (2 * jnp.pi)).astype(int)
    else:
        rs = jnp.concatenate([jnp.linspace(0, max_distance, n_points)[:, None], jnp.zeros((n_points, 2))], axis=-1)
        kpoints_int = jnp.concatenate([jnp.arange(-2, 3)[:, None], jnp.zeros((5, 2))], axis=-1).astype(int)
        # kpoints_int = jnp.rint(generate_kpoints_by_m(m=2) / (2 * jnp.pi)).astype(int)
    
    kpoints_names = np.array(['_'.join([str(i) for i in kpoint]) for kpoint in kpoints_int])
    kpoints = (kpoints_int * 2 * jnp.pi) @ mol.inv_basis

    exp_stats.kpoints = kpoints
    exp_stats.kpoints_int = kpoints_int
    
    e_idxs = [0,]
    t0 = time.time()
    for e_idx in e_idxs:
        _compute_pr = compute_prs[e_idx]

        for i, r in enumerate(rs):
            new_dims = (i for i in range(3-r.ndim))
            r_shift = jnp.expand_dims(r, axis=new_dims)

            prs = []
            for walkers in all_walkers:
                pr = jnp.mean(_compute_pr(walkers, r_shift))
                prs += [pr]
            pr = jnp.mean(jnp.array(prs))

            w = jnp.exp(1j * (kpoints @ r))
            exp_stats.nk = (w * pr)[None, :]

            t1 = time.time()
            print(f'Proportion complete r: {i/float(len(rs))} time {(t1 - t0):.0f}')
            t0 = t1

        exp_stats.process('nk', lambda x: np.mean(x, axis=0) / mol.volume)

    d = {}
    for k, v in exp_stats.get_dict().items():
        d[k] = np.array(v)
        if isinstance(v, jnp.ndarray):
            print('WHAT THE FUCK', v)
    save_pk(d, oj(run_dir, 'exp_stats_mom_dist.pk'))
    
    
if __name__ == '__main__':
    
    from utils import run_fn_with_sysargs
    args = run_fn_with_sysargs(compute)

    