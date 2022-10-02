
from nn_ansatz.utils import save_config_csv_and_pickle, save_pk
from nn_ansatz.sampling import equilibrate, keep_in_boundary
from utils import load_pk, oj, ojm
from nn_ansatz.ansatz import create_wf
from nn_ansatz.routines import initialise_system_wf_and_sampler
from nn_ansatz.systems import SystemAnsatz, generate_k_points
from nn_ansatz.plot import plot
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


def compute_gr(rs, distances, volume):
        '''
        http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html
        '''
        n_walkers = len(distances)
        n_ri, n_rj = distances.shape[1:]  # number of reference particles, number target
        number_density = n_rj / volume  # is the number density n_ri or 
        
        pdfs = []
        for i, r in enumerate(rs):
            r0 = rs[i]
            r1 = rs[i+1]
            counts = float(jnp.sum((r0<=distances)*(distances<=r1))) / n_ri
            volume_element = (4.*np.pi/3.) * (r1**3 - r0**3)
            # volume = 4.* np.pi * r0**2 * dr  # less accurate
            pdf = counts / (n_walkers * volume_element * number_density)
            pdfs.append(pdf)

        return np.array(pdfs)


def sample_sphere(shape, key, sphere_r, check=False):
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/

    subkeys = rnd.split(key, 3)

    theta = 2*jnp.pi * rnd.uniform(subkeys[0], shape)
    phi = jnp.arccos(1. - (2. * rnd.uniform(subkeys[1], shape)))

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


def cut_outliers(x: np.ndarray, y: np.ndarray, frac_of_data=0.95):
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


def generate_nd_hypercube(lims, n=20):
    points = [jnp.linspace(*lim, n) for lim in lims]
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


def compute_observables(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                        load_it: int=100000,
                        seed: int=0,
                        n_batch: int=10,
                        n_points: int=50,
                        plot_dir: str=None,
                        n_walkers_max: int=256,
                        walkers: Union[np.ndarray, None] = None,
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

    params_path = oj(models_path, load_file)
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, params_path=params_path, walkers_path=walkers_path)
    swf = create_wf(mol, signed=True)

    if walkers is not None:
        walkers = walkers[:n_walkers_max]

    walkers, step_size = equilibrate(params, 
                                        walkers, 
                                        keys, 
                                        mol=mol, 
                                        vwf=vwf,
                                        walkers_path=walkers_path,
                                        sampler=sampler, 
                                        compute_energy=True, 
                                        n_it=20000, 
                                        step_size_out=True)

    walkers = walkers[:n_walkers_max]
    print('Walkers shape ', walkers.shape)

    max_distance = (3*mol.scale_cell**2)**0.5
    dr = max_distance / n_points

    ncols, nrows = 3, 2
    figsize = get_fig_size(ncols, nrows) 
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    if isinstance(axs, Iterable): 
        axs = iter(axs.flatten())

    exp_stats = StatsCollector()

    ''' COMPUTE ONE BODY DENSITY MATRIX '''
    print('Computing one body density matrix')
    e_idxs = [0 if n_up == n_el else 0, 1]
    for e_idx in e_idxs:
        x_name = f'pr_x_{e_idx}'
        y_name = f'pr_{e_idx}'

        _compute_pr = jit(partial(compute_pr, swf=swf, e_idx=e_idx, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis))
        
        for sphere_r in np.linspace(0.0, max_distance, n_points):

            for i in range(10):
                key, sam_subkey = rnd.split(key, 2)
                walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)

            sphere_sample = sample_sphere((len(walkers),), key, sphere_r)
            exp_stats.set_by_name(y_name, jnp.mean(_compute_pr(walkers, sphere_sample)))
            exp_stats.set_by_name(x_name, sphere_r)
            
        pr_x, pr = cut_outliers(exp_stats.get(x_name), exp_stats.get(y_name))
        exp_stats.overwrite(y_name, pr)
        exp_stats.overwrite(x_name, pr_x)

        ax = next(axs)
        ax.plot(
            exp_stats.get(x_name),
            exp_stats.get(y_name),
        )
        format_ax(
            ax, 
            xlabel='r',
            ylabel='one body density p(r)',
        )

    ''' COMPUTE PAIR CORRELATION '''
    rs = jnp.linspace(0.0, max_distance, n_points)
    for nb in range(n_batch):
        
        for i in range(10):
            key, sam_subkey = rnd.split(key, 2)
            walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)

        walkers_up, walkers_down = split_spin_walkers(walkers, n_up)

        d_u = compute_distances(walkers_up, mol)
        d_d = compute_distances(walkers_down, mol)
        d_ud = compute_distances(walkers_up, mol, rj = walkers_down)

        exp_stats.gr_u = compute_gr(rs, d_u, mol.volume)[None, :]
        exp_stats.gr_d = compute_gr(rs, d_d, mol.volume)[None, :]
        exp_stats.gr_ud = compute_gr(rs, d_ud, mol.volume)[None, :]

    exp_stats.process(['gr_u', 'gr_d', 'gr_ud'], lambda x: np.mean(x, axis=0))

    ax = next(axs)
    labels = ['gr_u', 'gr_d', 'gr_ud']
    for l in labels:
        ax.plot(
            rs,
            exp_stats.get(l),
            label=l
        )
    format_ax(
        ax, 
        xlabel='r',
        ylabel='pair correlation g(r)',
        title='u=up, d=down',
        legend=True,
    )

    ''' COMPUTE MOMENTUM DISTRIBUTION '''
    print('Computing momentum distribution')
    rs = generate_nd_hypercube(([0., max_distance], [0., max_distance], [0., max_distance]), n=2)
    kpoints = generate_kpoints_by_m(m=2) @ mol.inv_basis
    kpoints_names = jnp.rint(((kpoints @ mol.basis) / (jnp.pi * 2))).astype(int)
    kpoints_names = [f'{i[0]:d}_{i[1]:d}_{i[2]:d}' for i in kpoints_names]

    kr = rs @ jnp.transpose(kpoints)  # (n_r, n_kpoints)
    wave = jnp.exp(1j * kr)

    e_idxs = [0,]
    for e_idx in e_idxs:
        _compute_pr = jit(partial(compute_pr, swf=swf, e_idx=e_idx, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis))

        for step, (r, w) in enumerate(zip(rs, wave)):
            key, sam_subkey = rnd.split(key, 2)
            new_dims = (i for i in range(3-r.ndim))
            r = jnp.expand_dims(r, axis=new_dims)
            
            for i in range(5):
                walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)
            
            pr = jnp.mean(_compute_pr(walkers, r))
            exp_stats.all_nk = (w * pr)[None, :]

        exp_stats.process('all_nk', lambda x: np.mean(x, axis=0) / mol.volume)

    idxs = np.argsort([np.linalg.norm(np.array([float(i) for i in k.split('_')]), axis=-1) for k in kpoints_names])
    x = np.arange(len(kpoints_names))
    y = exp_stats.all_nk[idxs]

    ax = next(axs)
    ax.plot(
        x, 
        y
    )

    format_ax(
        ax,
        xlabel='reciprocal space vector',
        ylabel='n_k',
        xticks=x[::5],
        xticklabels=np.array(kpoints_names)[idxs][::5],
        title='momentum all k'
    )

    norms = jnp.linalg.norm(kpoints, axis=-1)
    idxs = jnp.argsort(norms)
    x = norms[idxs]
    y = exp_stats.all_nk[idxs]

    ax = next(axs)
    ax.scatter(
        x=x,
        y=y,
    )
    format_ax(
        ax,
        xlabel='norm reciprocal space vector',
        ylabel='n_k',
        title='moment as function of norm(k)'
    )

    format_fig(
        fig,
        axs,
        fig_title=f'rs={mol.density_parameter} observables',
        fig_path=ojm(plot_dir, 'observables.png')
    )

    save_pk(exp_stats.get_dict(), oj(plot_dir, 'exp_stats.pk'))
    
    
if __name__ == '__main__':
    
    from utils import run_fn_with_sysargs
    args = run_fn_with_sysargs(compute_observables)


