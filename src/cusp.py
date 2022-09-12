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


'''
load in the configuration
set up the wave function
get samples
compute the quantity 
'''

SEED = 0


# def sample_sphere(shape, key, sphere_r=1e-5):
#     subkeys = rnd.split(key, 3)
#     theta = 2*jnp.pi * rnd.uniform(subkeys[0], shape)
#     phi = 1.-(2.*rnd.uniform(subkeys[1], shape))
    
#     # u = jnp.random.uniform(subkeys[2], shape)
#     # theta = jnp.arccos(costheta)
#     # r = sphere_r * jnp.cbrt(u)

#     x = sphere_r * jnp.sin( phi) * jnp.cos( theta )
#     y = sphere_r * jnp.sin( phi) * jnp.sin( theta )
#     z = sphere_r * jnp.cos( phi )

#     sphere_sample = jnp.stack([x, y, z], axis=-1)[:, None, :]
#     # r = jnp.sqrt(jnp.sum(sphere_sample**2, axis=-1, keepdims=True))
#     return sphere_sample


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


# f(r_ij) = \psi(r_i, r_i + s_ij, ...)

def compute_wf_rij(params, walkers, key, vwf, e0_idx, e1_idx, rij):
    
    n_el = walkers.shape[1]
    sphere_sample = sample_sphere((walkers.shape[0],), key, rij)
    mask = jnp.array([False if i != e1_idx else True for i in range(n_el)])[None, :, None]
    pos_e0 = walkers[:, e0_idx, :][:, None, :]
    pos_e1 = pos_e0 + sphere_sample            
    walkers_coalesce = jnp.where(mask, pos_e1, walkers)
    
    
    log_psi = vwf(params, walkers_coalesce)
    
    return jnp.sum(log_psi)
    

def compute_cusp_condition(run_dir: str='./experiments/HEG_PRX/bf_af_0/BfCs/seed0/run_0',
                           load_it: int=100000,
                           e_idxs: list[list[int]] = [[0,1],],
                           seed: int=0,
                           n_batch: int=10,
                           **kwargs         
    ):

    key = rnd.PRNGKey(seed)

    models_path = oj(run_dir, 'models')

    cfg = load_pk(oj(run_dir, 'config1.pk'))
    n_el, n_up = cfg['n_el'], cfg['n_up']

    params = oj(models_path, f'i{load_it}.pk')
    mol, vwf, walkers, params, sampler, keys = initialise_system_wf_and_sampler(cfg, walkers=None, load_params=params)
    swf = create_wf(mol, signed=True)
    
    n_walkers = walkers.shape[0] // cfg['n_devices']
    walkers = walkers[:n_walkers]
    walkers = jnp.repeat(walkers[0, ...][None, ...], n_walkers, axis=0)
    print('Walkers shape ', walkers.shape)

    ids = {k:'up' for k in range(cfg['n_up'])} | {k:'down' for k in range(cfg['n_up'], cfg['n_el'])}

    if e_idxs is None: 
        e_idxs = [[0, 1],] 
        if n_el != n_up: e_idxs.append([0, n_up])

    mode = 'resample'
    sphere_r_init = 1e-6
    key, subkey = rnd.split(key, 2)
    sphere_sample = sample_sphere((n_walkers,), subkey, sphere_r=sphere_r_init)
    norm_sphere_sample = sphere_sample / sphere_r_init

    results = {}
    for e0_idx, e1_idx in e_idxs:

        mask = jnp.array([False if i != e1_idx else True for i in range(n_el)])[None, :, None]
        # pos_e0 = jnp.repeat(walkers[:, e0_idx, :][:, None, :], n_el, axis=1)
        pos_e0 = walkers[:, e0_idx, :][:, None, :]

        exp_stats = {}
        for sphere_r in jnp.linspace(sphere_r_init, 1e-4, 20):
            # \frac{d \psi(X)}{d r_i} = \frac{d |\psi(X)|}{d r_i} * \mathrm{sign}(\psi(X)) 

            exp_stat = {}
            for _ in range(n_batch):
            
                key, subkey = rnd.split(key, 2)

                if mode == 'resample':
                    sphere_sample = sample_sphere((n_walkers,), subkey, sphere_r=sphere_r)
                    norm_sphere_sample = jnp.squeeze(sphere_sample / sphere_r)
                else:
                    exit('Potentially move single walkers in a straight line')

                pos_e1 = pos_e0 + sphere_sample
                
                walkers_coalesce = jnp.where(mask, pos_e1, walkers)
                walkers_equal = jnp.where(mask, pos_e0, walkers)

                log_psi_c, sign_c = swf(params, walkers_coalesce)
                psi_c = sign_c * jnp.exp(log_psi_c)
                log_psi_e, sign_e = swf(params, walkers_equal)
                psi_e = sign_e * jnp.exp(log_psi_e)

                gf_log_psi = grad(lambda w: jnp.sum(vwf(params, w)))
                g_log_psi = gf_log_psi(walkers_coalesce)[:, e1_idx, :]
                grij_log_psi = jnp.mean(jnp.inner(g_log_psi, norm_sphere_sample))

                gf_psi = grad(lambda w: jnp.sum(jnp.exp(vwf(params, w))))
                g_psi = gf_psi(walkers_coalesce)[:, e1_idx, :] 
                grij_psi = jnp.mean(jnp.squeeze(jnp.inner(g_psi, norm_sphere_sample)) * jnp.squeeze(sign_c))

                psip_psi = psi_c / psi_e

                wf_rij = lambda rij: compute_wf_rij(params, walkers, subkey, vwf, e0_idx, e1_idx, rij)
                log_psi_wf_rij = wf_rij(sphere_r)
                gf_wf_rij = grad(wf_rij)
                g_wf_rij = gf_wf_rij(sphere_r)
                
                exp_stat = append_dict_to_dict(exp_stat, 
                {
                            'r': float(sphere_r),
                            'log_psi': float(jnp.mean(log_psi_c)),
                            'g_log_psi': float(jnp.mean(g_log_psi)),
                            'grij_log_psi': float(jnp.mean(grij_log_psi)),
                            'psi': float(jnp.mean(psi_c)),
                            'g_psi': float(jnp.mean(g_psi)), 
                            'grij_psi': float(jnp.mean(grij_psi)),
                            'psip_psi': float(jnp.mean(psip_psi)),
                            'log_psi_wf_rij': float(jnp.mean(log_psi_wf_rij)),
                            'g_wf_rij': float(jnp.mean(g_wf_rij))
                })
            
            mean_exp_stat = {k: np.mean(np.array(v)) for k, v in exp_stat.items()}
            exp_stats = append_dict_to_dict(exp_stats, mean_exp_stat)
            
        exp_stats = {k:np.array(v) for k, v in exp_stats.items()}

        result_string = f'e{e0_idx}{ids[e0_idx]}_e{e1_idx}{ids[e1_idx]}_cusp'
        print('\n', result_string)
        # print('grads', grads)

        results[result_string] = float(grij_log_psi)

        # \Psi(rvec_1,rvec_1+xvec,rvec_3,....)/\Psi(rvec_1,rvec_1,rvec_3,....)
        exp_stats = pd.DataFrame.from_dict(exp_stats)
        exp_stats_name = oj(run_dir,f'{result_string}_stats.csv')
        exp_stats.to_csv(exp_stats_name)

        pretty_results_file = exp_stats_name[:exp_stats_name.rindex('.')]+'.txt'
        save_pretty_table(exp_stats.copy(), path=pretty_results_file)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(exp_stats['r'], exp_stats['log_psi'])
            result = 'log psi=mr + c: m ' + str(slope) + 'c' + str(intercept)
            append_to_txt(pretty_results_file, result)
        except Exception as e:
            print('fit failed log psi', e)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(exp_stats['r'], exp_stats['psi'])
            result = 'psi=mr + c: m ' + str(slope) + 'c' + str(intercept)
            append_to_txt(pretty_results_file, result)
        except Exception as e:
            print('fit failed psi', e)

    cfg.update(results)
    save_config_csv_and_pickle(cfg)

    print(walkers_coalesce[0])
    print(walkers_equal[0])


# get the mean of the gradients


if __name__ == '__main__':

    from utils import run_fn_with_sysargs

    args = run_fn_with_sysargs(compute_cusp_condition)

    # compute_cusp_condition(**args)



