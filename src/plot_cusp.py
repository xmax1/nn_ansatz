from turtle import Shape

from nn_ansatz.utils import save_config_csv_and_pickle, save_pk
from utils import append_dict_to_dict, collect_args, load_pk, oj, save_pretty_table, append_to_txt
from nn_ansatz.ansatz import create_wf
from nn_ansatz.routines import initialise_system_wf_and_sampler
from jax import random as rnd, numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cmaps = {'sequential': {x: plt.get_cmap(x) for x in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']}}

'''
load in the configuration
set up the wave function
get samples
compute the quantity 
'''

SEED = 0


def plot_surface(df, path):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    fig.savefig(path)


def plot_surface(data, path):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    cmap = cmaps['sequential']['viridis']

    n = int(np.sqrt(len(data['x'])))
    x = data['x'].reshape((n, n))
    y = data['y'].reshape((n, n))
    z = data['z'].reshape((n, n))

    surf = ax.plot_surface(x, y, z, 
                           cmap=cmap,
                           linewidth=1, 
                           antialiased=False)

    fig.colorbar(surf, shrink=0.4, aspect=10, pad=0.3) # Add a color bar which maps values to colors.
    fig.savefig(path)
        

def split_given_size(a, size):
    return jnp.split(a, jnp.arange(size, len(a), size))
    

def plot_cusp(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                           load_it: int=100000,
                           e_idxs: list[list[int]] = [[0, 7],],
                           seed: int=0,
                           n_batch: int=10,
                           n_point_side: int=10, 
                           n_walkers_max: int=512,
                           bounds: float=1e-5,
                           **kwargs         
    ):

    key = rnd.PRNGKey(seed)

    models_path = oj(run_dir, 'models')

    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']

    params = oj(models_path, f'i{load_it}.pk')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers=None, load_params=params)
    swf = create_wf(mol, signed=True)
    
    walkers = walkers[0, ...][None, ...]  # (1, n_el, 3)

    x = jnp.linspace(-1,1,n_point_side+1) * bounds
    y = jnp.linspace(-1,1,n_point_side+1) * bounds
    X, Y = jnp.meshgrid(x, y)
    x = X.ravel()
    y = Y.ravel()
    z = jnp.zeros((jnp.prod(jnp.array(x.shape)),))
    grid_sample = jnp.stack([x, y, z], axis=-1)[:, None, :]  # (n_point_sample**2, 1, 3)
    grids = split_given_size(grid_sample, n_walkers_max)

    exp_stats = {}
    for e0_idx, e1_idx in e_idxs:
        name = f'e{e0_idx}_e{e1_idx}'

        mask = jnp.array([False if i != e1_idx else True for i in range(n_el)])[None, :, None]  # (1, 14, 1)
        pos_e0 = walkers[:, e0_idx, :][:, None, :]  # (1, 1, 3)
        
        exp_stat = {}
        for grid in grids:
            
            pos_e1 = pos_e0 + grid  # (n_walkers_max, 1, 3)
            walkers_batch = jnp.where(mask, pos_e1, walkers)  # (1, 14, 1), (n_walkers_max, 1, 3), (1, n_el, 3)
            
            log_psi, sign = swf(params, walkers_batch)
            psi = sign * jnp.exp(log_psi)

            exp_stat = append_dict_to_dict(exp_stat,
                {
                    'r': np.squeeze(np.array(walkers_batch)),
                    'grid': np.squeeze(np.array(grid)),
                    'log_psi': np.squeeze(np.array(log_psi)),
                    'psi': np.squeeze(np.array(psi))
                }
            )

        exp_stats[name] = {k:np.squeeze(np.array(v)) for k, v in exp_stat.items()}
    
        exp_tmp = {
            'x': exp_stats[name]['grid'][:, 0], 
            'y': exp_stats[name]['grid'][:, 1], 
            'z': exp_stats[name]['psi']
        }
        
        plot_surface(exp_tmp, oj(kwargs['plot_dir'], f'cusp_{name}.svg'))

    save_pk(exp_stats, oj(kwargs['plot_dir'], 'results.pk'))
    

if __name__ == '__main__':

    from utils import run_fn_with_sysargs

    args = run_fn_with_sysargs(plot_cusp)

    # compute_cusp_condition(**args)



