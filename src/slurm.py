import itertools
from simple_slurm import Slurm  # https://pypi.org/project/simple-slurm/
import shutil
import yaml
import os
import string
from datetime import datetime
import pickle as pk
import pandas as pd

from utils import oj, mkdirs, save_pk_and_csv, load_yaml

#%%

'''
TODO run single file in all relevant experiments if a run not given

'''

# There was some issue finding the GPUs, if it comes back up, try loading these modules
#module load GCC
#module load CUDA/11.4.1
#module load cuDNN

N_MAX_ACTIVE_JOBS = 40
exp_dir = './experiments'
out_dir_tmp = f'{exp_dir}/out'
mkdirs(out_dir_tmp)

# puts the output files neatly. Trailing / ensures has defined directory
move_command = f'mv {out_dir_tmp}/o-$SLURM_JOB_ID.out {out_dir_tmp}/e-$SLURM_JOB_ID.err $out_dir' 

strip_characters = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
characters = [str(i) for i in range(10)]
characters.extend(list(string.ascii_lowercase))


def format_value_or_key(value):
    if isinstance(value, float):
        # takes 2 significant figures automatically
        return f'{value:.2g}'  
    elif isinstance(value, str):
        # removes '_', vowels (unless first), repeat letters, and capitalises first character
        value = value.replace('_', '')
        value = ''.join([x for i, x in enumerate(value) if (x not in strip_characters) or (i == 0)])
        value = ''.join(sorted(set(value), key=value.index))
        return value.capitalize()
    elif isinstance(value, int):
        # limit length of ints to 4
        return str(value)[:4]
    else:
        return str(value)


def gen_datetime():
    return datetime.now().strftime("%d%b%H%M")


def create_filename(cfg):
    cfg = {k:v for k,v in cfg.items() if 'seed' not in k}
    sorted_keys = sorted(cfg.keys())
    hyperparams_name = '_'.join([f'{format_value_or_key(k)}{format_value_or_key(cfg[k])}' for k in sorted_keys])
    if len(hyperparams_name) == 0: hyperparams_name = 'baseHypam'
    return hyperparams_name
    

def append_idx(root):
    idx = 0
    root_tmp = root + f'_{idx}'
    while os.path.exists(root_tmp):
        root_tmp = root + f'_{idx}'
        idx += 1
    return root_tmp


def create_dir_structure(root, exp):
        
    hp_name = create_filename(exp)
    exp['hypams_name'] = hp_name
    exp_dir = oj(f'{root}/{hp_name}/seed{exp["seed"]}')
    run_dir =  append_idx(oj(exp_dir, 'run'))
    exp_out = oj(run_dir, 'out')
    mkdirs(exp_out)
    
    save_pk_and_csv(exp, oj(f'{root}/{hp_name}', 'summary'))
    return run_dir


def boilerplate(env):
    cmd = f'module purge \n \
            source ~/.bashrc \n \
            module load GCC \n \
            module load CUDA/11.4.1 \n \
            module load cuDNN \n \
            conda activate {env} \n \
            pwd \n \
            nvidia-smi'
    return cmd


# creates a file of experiments
def write_exps_list(submission_cmds):
    with open(r'./exps.tmp', 'w') as fp:
        fp.write(' \n'.join(submission_cmds))
    return 


def make_slurm(time_h, n_cmds, submission_name):
    slurm = Slurm(
                mail_type='FAIL',
                partition='sm3090',
                N=1,  # n_node
                n=8,  # n_cpu
                time=f'0-{time_h}:00:00',
                output=f'{out_dir_tmp}/o-%j.out',
                error=f'{out_dir_tmp}/e-%j.err',
                gres='gpu:RTX3090:1',
                job_name=submission_name
            )
    if n_cmds > 1:
        # https://help.rc.ufl.edu/doc/SLURM_Job_Arrays %5 for max 5 jobs at a time
        slurm.add_arguments(array=f'0-{n_cmds-1}%{N_MAX_ACTIVE_JOBS}')  
    print(slurm)
    return slurm


def run_single_slurm(execution_file = 'cusp.py ',
                     submission_name = 'analysis',
                     cfg_file = 'sweep_cfg.yaml',
                     env = 'gpu',
                     time_h: int = 24,                     
                     **exp):

    '''
    In case 1 the experiment is new
    In case 2 we are doing further experiments on an existing model

    Case 1 needs to be covered - if the run_dir exists then we know it is case 2

    '''
    # Case 2 only
    run_dir = exp.get('run_dir')
    exp_name = exp.get('exp_name')
    if run_dir is None: exit('Not ready yet')
    if exp_name is None: exp_name = 'analysis'

    out_dir = append_idx(oj(run_dir, 'out', exp_name, 'out'))
    mkdirs(out_dir)
    
    plot_dir = oj(out_dir, 'plots')
    mkdirs(plot_dir)
    exp['plot_dir'] = plot_dir

    slurm = make_slurm(time_h, 1, submission_name)
    cmd = f'python -u {execution_file} ' # -u unbuffers print statements
    for k, v in exp.items():
        cmd += f' --{k} {str(v)} '

    print(cmd)

    slurm.sbatch(f'{boilerplate(env)} \n \
                   out_dir={out_dir} \n \
                   {cmd} | tee $out_dir/py.out \n \
                   {move_command} \n \
                   date "+%B %V %T.%3N"'
                )
    return 


def run_slurm_sweep(execution_file = 'run_with_args.py ',
                    submission_name = 'sweep',
                    cfg_file = 'sweep_cfg.yaml',
                    use_array = True,
                    env = 'dw',
                    exp_name = None,
                    time_h = 24,
                    ):

    sweep_cfg = load_yaml(cfg_file)

    if exp_name is None: exp_name = sweep_cfg['base_cfg'].get('exp_name') 
    if exp_name is not None: exp_dir = append_idx(oj(exp_dir, exp_name))

    sweep_hp = list(itertools.product(*sweep_cfg['sweep'].values()))

    base_cmd = f'python -u {execution_file} '  # -u unbuffers print statements
    for k, v in sweep_cfg['base_cfg'].items():
        base_cmd += f' --{k} {str(v)} '

    sub_cmds = []
    hp_names = []
    for hp in sweep_hp:
        
        sub_cmd = str(base_cmd)
        exp = {k: v for k, v in zip(sweep_cfg['sweep'].keys(), hp)}
        
        for k, v in exp.items():
            sub_cmd += f' --{k} {str(v)} '

        run_dir = create_dir_structure(exp_dir, exp)
        out_dir = oj(run_dir, 'out')
        hp_name = create_filename(exp)
        hp_names.append(hp_name)

        sub_cmd += f' --run_dir {run_dir} --out_dir {out_dir}'
        sub_cmds.append(sub_cmd)  # for array
        # End: Creating the file structure

    assert not any([hp_names.count(x) == 1 for x in hp_names])  # checks none of the exp names are the same

    if use_array:

        write_exps_list(sub_cmds)
        n_cmds = len(sub_cmds)

        slurm = make_slurm(time_h, n_cmds, submission_name)

        print(slurm)

        '''
        put the exp file into array
        remove other modules
        get conda paths and cuda / cudnn (if gpu disappears)
        activate the conda env
        print the working dir
        make the output dirs
        run the cmd
        move outputs to output dir
        double curly braces in the f-string prints a curly brace
        '''

        slurm.sbatch(f'mapfile -t myArray < exps.tmp \n \
                       cmd=${{myArray[$SLURM_ARRAY_TASK_ID]}} \n \
                       out_dir=${{cmd#*out_dir}} \n \
                       out_dir=$(echo ${{out_dir}} | tr -d " ") \n \
                       {boilerplate(env)} \n \
                       $cmd | tee $out_dir/py.out \n \
                       {move_command} \n \
                       date "+%B %V %T.%3N" '
                    )

    shutil.copyfile(cfg_file, f'{exp_dir}/{cfg_file}')

'''
TODO --depend=afterany:{$SLURM_ARRAY_JOB_ID}  # https://slurm.schedmd.com/job_array.html
TODO strip whitespace #    run_dir=$run_dir | tr -d ' ' \n \

'''

if __name__ == '__main__':
    from utils import collect_args, type_args
    from itertools import combinations

    args = collect_args()
    typed_args = type_args(args, run_single_slurm)

    root = '/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/final1001/14el/baseline/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s128_p32_l3_det1/'

    exe_files = ['obs_pair_corr.py', 'obs_one_body.py', 'obs_mom_dist.py']

    test_path = 'run41035'
    paths = list(os.listdir(root))
    # paths.remove(test_path)
    # paths = [test_path,]
    for path in paths:
        for exe_file in exe_files:
            exp = {
                'run_dir': os.path.join(root, path),
                'd3': False,
                'n_points': 25,
                'n_walkers_max': 1024,
                'n_walkers': None,
                'execution_file': exe_file,
            }
            exp['exp_name'] = exe_file.split('.')[0] + f'_d3{str(exp["d3"])}' + f'_nw{exp["n_walkers"]}'

            exp = exp | typed_args
            if 'pair_corr' in exe_file:
                exp['n_walkers'] = None

            run_single_slurm(**exp, time_h=24)
        

    # python slurm.py --execution_file compute_observables.py --exp_name 100batch_d1 --n_batch 1000 --n_points 25 --d3 False
    # python slurm.py --execution_file compute_observables.py --exp_name 1006/100b --n_batch 100 --n_points 25 --d3 False
    # python slurm.py --execution_file compute_observables.py --exp_name 100batch_d3 --n_batch 100 --n_points 25 --d3 True
    # python slurm.py --execution_file compute_observables.py --exp_name 10batch_d3 --n_batch 10 --n_points 25 --d3 True

    # python slurm.py --execution_file compute_observables.py --exp_name 10batch_xyz --n_batch 10 --n_points 25
    # python slurm.py --execution_file compute_observables.py --exp_name gen_walkers
    
        



