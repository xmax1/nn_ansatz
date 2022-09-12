from turtle import Shape

from nn_ansatz.utils import save_config_csv_and_pickle, save_pk
from utils import append_dict_to_dict, collect_args, load_pk, oj
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
        rj_walkers = jnp.copy(walkers)
    else:
        rj_walkers = rj

    displacement = walkers[..., None, :] - jnp.expand_dims(rj_walkers, axis=-3)
    displacement = apply_minimum_image_convention(displacement, mol.basis, mol.inv_basis)
    distances = jnp.linalg.norm(displacement, axis=-1)
    return distances  # (n_walkers, n_i, n_j)


def split_spin_walkers(walkers, n_up):
    walkers_up = walkers[..., :n_up, :]
    if n_up == walkers.shape[-2]:
        walkers_down = None
    else:
        walkers_down = walkers[..., n_up:, :]
    return walkers_up, walkers_down



def compute_gr(distances, n_bins=100):
        n_walkers = len(distances)
        distances = distances.reshape(-1)
        max_distance = max(distances)
        n_el_target = jnp.prod(jnp.array(distances.shape[1:]))

        dr = max_distance/float(n_bins)
        rs = np.linspace(0.000001, max_distance, n_bins)  # 0.000001 cuts off zeros
        
        pdfs = []
        for r in rs:
            outer_edge = r + dr
            counts = float(np.sum((r<distances)*(distances<outer_edge)))
            volume = (4.*np.pi/3.) * (outer_edge**3 - r**3)
            pdf = counts / (n_walkers * volume)
            pdfs.append(pdf)
        return rs, np.array(pdfs)
    


def compute_pair_correlation(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                             load_it: int=100000,
                             seed: int=0,
                             n_batch: int=10,
                             n_points: int=100,
                             plot_dir: str=None,
                             **exp         
    ):

    key = rnd.PRNGKey(seed)

    models_path = oj(run_dir, 'models')

    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']
    n_devices = cfg['n_devices']
    cfg['n_devices'] = 1
    print('n_devices = ', n_devices)

    params_path = oj(models_path, f'i{load_it}.pk')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, params_path=params_path)
    
    n_walkers = walkers.shape[0] // n_devices
    walkers = walkers[:n_walkers]
    print('Walkers shape ', walkers.shape)

    step_size = mol.step_size

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
    
    rs_ud, up_down_gr = compute_gr(exp_stats['up_down_d'], n_bins=n_points)
    rs_uu, up_up_gr = compute_gr(exp_stats['up_d'], n_bins=n_points)
    rs_dd, down_down_gr = compute_gr(exp_stats['down_d'], n_bins=n_points)
    
    xs = [rs_uu, rs_dd, rs_ud]
    ys = [up_up_gr, down_down_gr, up_down_gr]
    titles = ['gr_up_up', 'gr_down_down', 'gr_up_down']
    ylabels = ['electron density g(r)'] * 3
    xlabels = ['r'] * 3
    
    plot(
        xs, 
        ys, 
        xlabels=xlabels, 
        ylabels=ylabels, 
        titles=titles,
        marker=None,
        linestyle='-',
        fig_title='gr',
        fig_path=oj(plot_dir, 'gr.png')
    )
    
    exp_stats = exp_stats | {'r_up_up': rs_uu, 'r_down_down': rs_dd, 'r_up_down': rs_ud}

    save_pk(exp_stats, oj(plot_dir, 'samples.pk'))
    
    




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



