from turtle import Shape

from nn_ansatz.utils import save_config_csv_and_pickle
from utils import append_dict_to_dict, collect_args, load_pk, oj, save_pretty_table, append_to_txt
from nn_ansatz.ansatz import create_wf
from nn_ansatz.routines import initialise_system_wf_and_sampler
from jax import random as rnd, numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn_ansatz.plot import plot
from nn_ansatz.vmc import create_energy_fn


def split_given_size(a, size):
    return jnp.split(a, jnp.arange(size, len(a), size))


def cusp_line(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                           load_it: int=100000,
                           e_idxs: list[list[int]] = [[0,1],],
                           seed: int=0,
                           n_points: int=1000,
                           n_walkers_max: int=512,
                           plot_dir: str=None,
                           **kwargs         
    ):

    key = rnd.PRNGKey(seed)


    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']

    models_path = oj(run_dir, 'models')
    params_path = oj(models_path, f'i{load_it}.pk')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, params_path=params_path)
    swf = create_wf(mol, signed=True)
    energy_function = create_energy_fn(mol, vwf, separate=True)
    pe, ke = energy_function(params, walkers)
    print('es', jnp.mean(pe), jnp.mean(ke))

    log_psi = vwf(params, walkers)
    xs = [np.squeeze(np.array(log_psi))]
    plot(xs, xlabels=['log psi'], plot_type='hist', fig_title='histogram log psi initial walkers')
    
    walkers = walkers[0, :, :][None, ...]  # (1, n_el, 3)

    print('Walkers shape ', walkers.shape)

    ids = {k:'up' for k in range(cfg['n_up'])} | {k:'down' for k in range(cfg['n_up'], cfg['n_el'])}

    if e_idxs is None: 
        e_idxs = [[0, 1],] 
        if n_el != n_up: e_idxs.append([0, n_up])

    grid = jnp.concatenate([jnp.linspace(-0.5, 0.5, n_points)[:, None, None], jnp.zeros((n_points, 1, 2))], axis=-1)  # (n_points, 1, 3)

    grids = split_given_size(grid, n_walkers_max)

    for e0_idx, e1_idx in e_idxs:
        result_string = f'e{e0_idx}{ids[e0_idx]}_e{e1_idx}{ids[e1_idx]}_cusp'
        print('\n', result_string)

        mask = jnp.array([False if i != e1_idx else True for i in range(n_el)])[None, :, None]
        pos_e0 = walkers[:, e0_idx, :][:, None, :]

        exp_stats = {}
        for grid in grids:  
            # \frac{d \psi(X)}{d r_i} = \frac{d |\psi(X)|}{d r_i} * \mathrm{sign}(\psi(X)) 

            pos_e1 = pos_e0 + grid  # (n_g, 1, 3) = (1, 1, 3) + (n_g, 1, 3)
            
            walkers_coalesce = jnp.where(mask, pos_e1, walkers)

            log_psi_c, sign_c = swf(params, walkers_coalesce)
            psi_c = sign_c * jnp.exp(log_psi_c)

            gf_log_psi = grad(lambda w: jnp.sum(vwf(params, w)))
            gx_log_psi = gf_log_psi(walkers_coalesce)[:, e1_idx, 0]

            gf_psi = grad(lambda w: jnp.sum(jnp.exp(vwf(params, w))))
            gx_psi = gf_psi(walkers_coalesce)[:, e1_idx, 0]

            pe, ke = energy_function(params, walkers_coalesce)
            
            exp_stats = append_dict_to_dict(exp_stats, 
            {k: np.squeeze(np.array(v)) for k, v in 
                {
                        'dx': grid[:, 0, 0],
                        'log_psi': log_psi_c,
                        'gx_log_psi': gx_log_psi,
                        'psi': psi_c,
                        'gx_psi': gx_psi,
                        'pe': pe,
                        'ke': ke
                }.items()
            })

        exp_stats = pd.DataFrame.from_dict(exp_stats)
 
        exp_stats_name = oj(plot_dir, f'{result_string}_stats.csv')
        exp_stats.to_csv(exp_stats_name)

        pretty_results_file = exp_stats_name[:exp_stats_name.rindex('.')]+'.txt'
        save_pretty_table(exp_stats.copy(), path=pretty_results_file)

        ylabels = ['log_psi', 'gx_log_psi', 'psi', 'gx_psi', 'pe', 'ke']
        xs = [exp_stats['dx']] * len(ylabels)
        ys = [exp_stats[y] for y in ylabels]
        fig = plot(
            xs, 
            ys, 
            xlabels=['dx']*len(ylabels), 
            ylabels=ylabels,
            marker='.', 
            linestyle=None,
            fig_title=result_string,
            fig_path=oj(plot_dir, result_string + '.png')
        )
        
        


if __name__ == '__main__':

    from utils import run_fn_with_sysargs

    args = run_fn_with_sysargs(cusp_line)

    # compute_cusp_condition(**args)



