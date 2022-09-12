
from enum import unique
import os
import pickle as pk
import pandas as pd
import shutil
import numpy as np
import jax.numpy as jnp

# import sys
# sys.path.append('../../nn_ansatz')

from nn_ansatz.utils import load_pk, oj, dict_append

def summarise(exp_name = './experiments/HEG_PRX/bf_af_0',
              seed_summary_file = 'summary_seed.csv'):
    
    summary_variables = []

    exps = os.listdir(exp_name)
    walk = os.walk(exp_name)

    # SUMMARISE ACROSS SEEDS
    exp_roots = []  # Base directories for different seed runs
    for root, dirs, files in walk:
        if 'seed1' in dirs:
            exp_roots.append(root)

    summary = {}  # Go to all the seed roots
    for exp_root in exp_roots:
        walk = os.walk(exp_root)
        
        config_paths = []  # Get the config paths
        for root, _, files in walk:
            if any([(('config' in f) and ('pk' in f)) for f in files]):
                config_pk_name = [x for x in files if (('config' in x) and ('pk' in x))][0]
                config_path = oj(root, config_pk_name)
                config_paths.append(config_path)
        
        # Compute statistics
        df = pd.DataFrame.from_dict([load_pk(p) for p in config_paths])
        new_csv_path = oj(exp_root, 'all_config.csv')
        df.to_csv(new_csv_path)
        df = pd.read_csv(new_csv_path)

        if len(summary_variables) == 0:
            df_vars = df.select_dtypes(include=[np.float64, np.float32, float])  # does not work on jnp / objects 
        else:
            df_vars = df[summary_variables]
        
        # Extract the config across seeds
        nunique = df_vars.nunique()
        unique_columns = nunique[(nunique != 1)].index
        
        df_vars = df_vars[unique_columns]
        exp_cfg = df.drop(unique_columns, axis=1).iloc[0]
        
        # compute the means / compute the stds
        means = df_vars.mean()
        stds = df_vars.std()
        stds.rename(index = {c:c+'_seedstd' for c in stds.index}, inplace = True)
        df = pd.DataFrame(pd.concat([exp_cfg, means, stds], axis=0)).transpose()  # 0 for index, 1 for columns, these are Series so 0
        df.to_csv(oj(exp_root, seed_summary_file))  # first column is named index

    # SUMMARISE ACROSS EXPERIMENTS
    # must use index_col to read in the first column as index
    df = pd.concat([pd.read_csv(oj(exp_root, seed_summary_file)) for exp_root in exp_roots], axis=0)
    df.to_csv(oj(exp_name, 'exp_summary.csv'))

if __name__ == '__main__':
    summarise()