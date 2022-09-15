from turtle import Shape

import os
from nn_ansatz.utils import save_config_csv_and_pickle, save_pk
from nn_ansatz.sampling import equilibrate
from utils import append_dict_to_dict, collect_args, load_pk, oj, ojm
from nn_ansatz.ansatz import create_wf
from nn_ansatz.routines import initialise_system_wf_and_sampler
from jax import random as rnd, numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from nn_ansatz.plot import plot
from nn_ansatz.ansatz_base import apply_minimum_image_convention


def compute_distances(walkers, mol, rj=None):
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



def compute_gr(dr, distances, volume_cell, r0=1e-6):
        '''
        http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html
        '''
        n_walkers = len(distances)
        max_distance = np.max(distances)
        n_ri, n_rj = distances.shape[1:]  # number of reference particles, number target
        number_density = n_rj / volume_cell  # is the number density n_ri or 
        
        rs, pdfs = [], []
        while r0 < (max_distance+dr):
            r_dr = r0+dr
            
            counts = float(np.sum((r0<=distances)*(distances<=r_dr))) / n_ri
            # volume = (4.*np.pi/3.) * (outer_edge**3 - r**3)
            volume = 4.* np.pi * r0**2 * dr  # this is approximate, why not use exact?
            pdf = counts / (n_walkers * volume * number_density)

            r0 = r_dr
            rs.append(r0 - dr/2.)
            pdfs.append(pdf)

        return rs, np.array(pdfs)
    


def compute_pair_correlation(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                             load_it: int=100000,
                             seed: int=0,
                             n_batch: int=10,
                             n_points: int=100,
                             plot_dir: str=None,
                             n_walkers_max: int=256,
                             **exp         
    ):

    key = rnd.PRNGKey(seed)

    models_path = oj(run_dir, 'models')
    walkers_dir = oj(run_dir, 'walkers')
    load_file = f'i{load_it}.pk'

    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']

    params_path = oj(models_path, load_file)
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, params_path=params_path)
    walkers = walkers[:n_walkers_max]
    
    walkers_dir = oj(run_dir, 'walkers')
    walkers_path = ojm(walkers_dir, load_file)
    
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

    exp_stats = {}
    for nb in range(n_batch):
        '''
        - sample
        - compute distances
        - put into histogram
            - what is the size of the box? 
            - We are in r_s units, is the cell changed? 
            - I'm not sure we are in r_s units... wHaT
            - split into up and down pair correlation? 
        '''
        key, sam_subkey = rnd.split(key, 2)

        for i in range(100):
            walkers, acc, step_size = sampler(params, walkers, sam_subkey, step_size)

        walkers_up, walkers_down = split_spin_walkers(walkers, n_up)
        
        up_d = compute_distances(walkers_up, mol)  # (n_walkers,)
        down_d = compute_distances(walkers_down, mol)
        up_down_d = compute_distances(walkers_up, mol, rj=walkers_down)

        exp_stat = {
            'up_d': np.squeeze(np.array(up_d)),
            'down_d': np.squeeze(np.array(down_d)),
            'up_down_d': np.squeeze(np.array(up_down_d)),
            'walkers': np.squeeze(np.array(walkers))
        }

        exp_stats = append_dict_to_dict(exp_stats, exp_stat)

    # n_max = max([len(v) for v in exp_stats.values()])
    # exp_stats = {k: np.concatenate([v, np.full((n_max-len(v), v.shape[1], v.shape[2]), np.nan)]) for k, v in exp_stats.items()}
    
    max_distance = (3*mol.scale_cell**2)**0.5
    # max_distance = mol.scale_cell
    print((exp_stats['up_down_d'] <= max_distance).all(), \
          (exp_stats['up_d'] <= max_distance).all(), \
          (exp_stats['down_d'] <= max_distance).all())
    print('This is not necessarily true, because there are the corners of the box')
    # when are they counted twice? 
    # an arc at the edge of the cube 

    dr = max_distance / n_points
    rs_ud, up_down_gr = compute_gr(dr, exp_stats['up_down_d'], mol.volume)
    rs_uu, up_up_gr = compute_gr(dr, exp_stats['up_d'], mol.volume)
    rs_dd, down_down_gr = compute_gr(dr, exp_stats['down_d'], mol.volume)
    
    plot(
        xdata=[rs_uu, rs_dd, rs_ud], 
        ydata=[up_up_gr, down_down_gr, up_down_gr], 
        xlabel='r', 
        ylabel='pair correlation g(r)', 
        title=['gr_up_up', 'gr_down_down', 'gr_up_down'],
        vlines=mol.scale_cell, 
        marker=None,
        linestyle='-',
        fig_title='gr',
        fig_path=oj(plot_dir, 'gr.png'),
    )
    
    exp_stats = exp_stats | {'r_up_up': rs_uu, 'r_down_down': rs_dd, 'r_up_down': rs_ud}

    save_pk(exp_stats, oj(plot_dir, 'exp_stats.pk'))
    
    print('n_walkers total ', len(exp_stats['up_down_d']))




    # exp_stats = pd.DataFrame.from_dict(exp_stats)
    
    # hist_data_long = exp_stats.melt(var_name='spin_spin', value_name='r')
    # grid = sns.displot(data=hist_data_long, col='spin_spin', x='r', kind='kde')
    
    # grid.fig.suptitle('Pair correlation g(r)')
    # grid.fig.subplots_adjust(top=.85)
    # grid.fig.savefig(oj(exp['plot_dir'], 'gr.png'))
    
    # max_range = np.sqrt(3*cfg['scale_cell']**2)
    # hists = {k:np.histogram(v, bins=100, range=(0., max_range)) for k, v in exp_stats.items()}


if __name__ == '__main__':

    from utils import run_fn_with_sysargs

    args = run_fn_with_sysargs(compute_pair_correlation)



