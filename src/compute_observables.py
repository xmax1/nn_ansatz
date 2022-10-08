
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


def generate_equilibrated_walkers(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                                  load_it: int=100000,
                                  seed: int=0,
                                  n_batch: int=10,
                                  n_points: int=50,
                                  plot_dir: str=None,
                                  n_walkers_max: int=256,
                                  walkers: Union[np.ndarray, None] = None,
                                  d3: bool = True,
                                  n_check: int=5,
                                  n_check_batchsize: int=128,
                                  **exp):

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
    mol, vwf, walkers, params, sampler, key = initialise_system_wf_and_sampler(cfg, params_path=params_path, walkers_path=walkers_path)
    swf = jit(create_wf(mol, signed=True))

    if walkers is not None:
        walkers = walkers[:n_walkers_max]

    print(walkers.shape) if walkers is not None else print(walkers)
    compute_energy = create_energy_fn(mol, vwf, separate=True)

    walkers, step_size = equilibrate(params, 
                                     walkers, 
                                     key, 
                                     mol=mol, 
                                     vwf=vwf,
                                     walkers_path=walkers_path,
                                     sampler=sampler, 
                                     compute_energy=False, 
                                     n_it=20000, 
                                     step_size_out=True)

    n_walkers_init = n_walkers_max
    n_walkers_max = {
        14: 256,
    }[mol.n_el]
    
    
    
    print(f'Trying n_walkers_max {n_walkers_max}')
    
    sf = n_walkers_max // n_walkers_init
    if n_walkers_max > n_walkers_init:
        walkers = jnp.repeat(walkers, int(sf), axis=0)
    else:
        walkers = walkers[:n_walkers_max]

    ke, pe = compute_energy(params, walkers)
    mean_ke, std_ke = jnp.mean(ke), jnp.std(ke)
    mean_pe, std_pe = jnp.mean(pe), jnp.std(pe)
    print('Initial energy')
    print(f'KE e_mean {mean_ke} e_std {std_ke}')
    print(f'PE e_mean {mean_pe} e_std {std_pe}')

    n_batch = int(1000000 // n_walkers_max) + 1
    checklist = np.arange(2, n_batch, n_batch//n_check).astype(int)
    t0 = time.time()
    all_walkers = walkers
    for b in range(n_batch):
        for i in range(10):
            key, subkey = rnd.split(key, 2)
            walkers, acc, step_size = sampler(params, walkers, subkey, step_size)

        all_walkers = jnp.concatenate([all_walkers, walkers], axis=0)
        walkers = all_walkers[rnd.randint(subkey, (n_walkers_max, ), 0, len(all_walkers)-1)]
        if b in checklist:
            print(f'{(b+1) * len(walkers)} generated... time {time.time() - t0:.2f}')
            ke, pe = compute_energy(params, walkers)
            mean_ke, std_ke = jnp.mean(ke), jnp.std(ke)
            mean_pe, std_pe = jnp.mean(pe), jnp.std(pe)
            print(f'KE e_mean {mean_ke} e_std {std_ke}')
            print(f'PE e_mean {mean_pe} e_std {std_pe}')
    
    save_pk(np.array(all_walkers), ojm(run_dir, f'eq_walkers_i{load_it}.pk'))


def compute_observables(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                        load_it: int=100000,
                        seed: int=0,
                        n_batch: int=10,
                        n_points: int=50,
                        plot_dir: str=None,
                        n_walkers_max: int=256,
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
    max_distance = mol.scale_cell
    print('Max distance: ', max_distance, 'Scale: ', mol.scale_cell)

    ncols, nrows = 3, 2
    figsize = get_fig_size(ncols, nrows) 
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    if isinstance(axs, Iterable): 
        axs = iter(axs.flatten())

    compute_prs = {
        0: jit(partial(compute_pr, swf=swf, e_idx=0, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis)),
        7: jit(partial(compute_pr, swf=swf, e_idx=7, n_el=n_el, params=params, basis=mol.basis, inv_basis=mol.inv_basis))
    }
    n_steps = 10
    _sample_sphere = jit(partial(sample_sphere, n_walkers=n_walkers_max))
    _compute_distances = jit(partial(compute_distances, mol=mol))
    _compute_gr = jit(partial(compute_gr, volume=mol.volume, n_points=n_points))

    exp_stats = StatsCollector()

    ''' COMPUTE ONE BODY DENSITY MATRIX '''
    print('Computing one body density matrix')
    e_idxs = [0 if n_up == n_el else 0, 7]
    for e_idx in e_idxs:
        
        _compute_pr = compute_prs[e_idx]
        x_name = f'pr_x_{e_idx}'
        y_name = f'pr_{e_idx}'
        
        for sphere_r in np.linspace(0.0, max_distance, n_points):
            
            prs = []
            for n in range(n_batch):
                for i in range(n_steps):
                    key, sam_subkey = rnd.split(key, 2)
                    walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)

                key, sam_subkey = rnd.split(key, 2)
                if d3:
                    sphere_sample = _sample_sphere(sam_subkey, sphere_r)
                else:
                    sphere_sample = jnp.array([[[sphere_r, 0.0, 0.0]]])
                    
                pr = jnp.mean(_compute_pr(walkers, sphere_sample))
                prs.append(pr)
            
            pr = jnp.mean(jnp.array(prs))

            exp_stats.set_by_name(x_name, sphere_r)
            exp_stats.set_by_name(y_name, pr)
        
            # exp_stats.process(x_name, lambda x: np.mean(x, axis=0, keepdims=True))
            # exp_stats.process(y_name, lambda x: np.mean(x, axis=0, keepdims=True))

        pr_x, pr = cut_outliers(exp_stats.get(x_name), exp_stats.get(y_name), passthrough=True)
        exp_stats.overwrite(x_name, pr_x)
        exp_stats.overwrite(y_name, pr)

        x, y = exp_stats.get(x_name), exp_stats.get(y_name)
        
        ax = next(axs)
        ax.plot(
            x,
            y,
        )
        format_ax(
            ax, 
            xlabel=x_name,
            ylabel='one body density p(r)',
        )

    ''' COMPUTE PAIR CORRELATION '''
    rs = jnp.linspace(0.0, max_distance, n_points)
    for nb in range(n_batch):
        
        for i in range(n_steps):
            key, sam_subkey = rnd.split(key, 2)
            walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)

        walkers_up, walkers_down = split_spin_walkers(walkers, n_up)

        d_u = _compute_distances(walkers_up)
        d_d = _compute_distances(walkers_down)
        d_ud = _compute_distances(walkers_up, rj = walkers_down)

        exp_stats.gr_u = _compute_gr(rs, d_u)[None, :]
        exp_stats.gr_d = _compute_gr(rs, d_d)[None, :]
        exp_stats.gr_ud = _compute_gr(rs, d_ud)[None, :]

    exp_stats.process(['gr_u', 'gr_d', 'gr_ud'], lambda x: np.mean(x, axis=0))

    exp_stats.rs = rs[:-1] + (rs[1:] - rs[:-1])/2

    ax = next(axs)
    labels = ['gr_u', 'gr_d', 'gr_ud']
    for l in labels:
        ax.plot(
            exp_stats.rs[5:],
            exp_stats.get(l)[5:],
            label=l
        )
    ax.set_xlim(left=0.0)

    format_ax(
        ax, 
        xlabel='r',
        ylabel='pair correlation g(r)',
        title='u=up, d=down',
        legend=True,
    )

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

    print(kpoints_int.shape)
    
    e_idxs = [0,]
    for e_idx in e_idxs:
        _compute_pr = compute_prs[e_idx]

        for r in rs:
            new_dims = (i for i in range(3-r.ndim))
            r_shift = jnp.expand_dims(r, axis=new_dims)

            prs = []
            for n in range(n_batch):
                for i in range(n_steps):
                    key, sam_subkey = rnd.split(key, 2)
                    walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)
                
                pr = jnp.mean(_compute_pr(walkers, r_shift))
                prs += [pr]
            
            prs = jnp.mean(jnp.array(prs))

            w = jnp.exp(1j * (kpoints @ r))
            exp_stats.all_nk = (w * prs)[None, :]

        exp_stats.process('all_nk', lambda x: np.mean(x, axis=0) / mol.volume)

    # kpoints_norm = jnp.linalg.norm(kpoints, axis=-1)
    # idxs = jnp.argsort(kpoints_norm)
    
    x = np.arange(len(kpoints))
    y = exp_stats.all_nk

    print(x.shape, y.shape)
    
    exp_stats.mom_dist_x = x
    exp_stats.mom_dist = y
    
    exp_stats.mom_dist_xticklabels = kpoints_names

    ax = next(axs)
    ax.plot(
        x, 
        y
    )

    format_ax(
        ax,
        xlabel='reciprocal space vector',
        ylabel='n_k',
        xticks=x[::10],
        xticklabels=exp_stats.mom_dist_xticklabels[::10],
        title='momentum all k'
    )

    norms = jnp.linalg.norm(kpoints, axis=-1)
    idxs = jnp.argsort(norms)
    x = norms[idxs]
    y = exp_stats.all_nk[idxs]

    exp_stats.mom_dist_norm_x = x
    exp_stats.mom_dist_norm = y

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

    d = {}
    for k, v in exp_stats.get_dict().items():
        d[k] = np.array(v)
        if isinstance(v, jnp.ndarray):
            print('WHAT THE FUCK', v) 

    save_pk(d, oj(plot_dir, 'exp_stats.pk'))
    
    
if __name__ == '__main__':
    
    from utils import run_fn_with_sysargs
    
    # args = run_fn_with_sysargs(compute_observables)
    args = run_fn_with_sysargs(generate_equilibrated_walkers)

    

    # compute_observables(run_dir = run_dir,
    #                     load_it = 100000,
    #                     seed = 0,
    #                     n_batch = 2,
    #                     n_points =5,
    #                     plot_dir = plot_dir,
    #                     n_walkers_max = 8,
    #                     walkers = None)


