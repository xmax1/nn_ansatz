
import datetime
from operator import methodcaller
import os
import pickle as pk
from torch.utils.tensorboard import SummaryWriter
import time
from jax import jit
from jax.experimental import optimizers
import tensorflow as tf
import jax
import jax.random as rnd
import jax.numpy as jnp
import toml
import glob
import os
import pandas as pd
from jax.tree_util import tree_flatten
import numpy as np
import csv
import sys

from .python_helpers import *


PATH = os.path.abspath(os.path.dirname(__file__))
systems_data = toml.load(os.path.join(PATH, 'systems_data.toml'))


def split_variables_for_pmap(n_devices, *args):
    if bool(os.environ.get('DISTRIBUTE')) is True:
        new_args = []
        for arg in args:
            if type(arg) in (float, int):
                new_arg = jnp.array(arg).repeat(n_devices)
            else:
                print('argument for splitting is type %s' % str(type(arg)))
            new_args.append(new_arg)
        
        if len(new_args) == 1:  # needed to unpack the list if there is just one element
            return new_args[0]
        return new_args
    if len(args) == 1:
        return args[0]
    return args

@jit
def key_gen(keys):
    """
    keys: (n_devices, 2)
    Pseudocode:
        - generate the new keys for each device and put into a new array
        - split the array along axis 1 so there are 2 arrays (keys and subkeys)
        - squeeze the middle axis out
    """
    if len(keys.shape) == 2:  # if distributed
        keys = jnp.array([rnd.split(key) for key in keys])
        keys = jnp.split(keys, 2, axis=1)
        return [x.squeeze(axis=1) for x in keys]
    return rnd.split(keys)

# key_gen = lambda keys: [x.squeeze() for x in jnp.array([rnd.split(key) for key in keys]).split(2, axis=1)]


def compare(tc_arr, jnp_arr):
    arr = tc_arr.detach().cpu().numpy()
    diff = arr - jnp_arr
    print('l1-norm: ', jnp.mean(jnp.abs(diff)))


def remove_aux(fn, axis=0):
    # creates a new function only outputting returns up to axis
    def _new_fn(*args):
        return fn(*args)[:(axis+1)]
    return _new_fn


def n2n(number, identifier=''):
    if type(number) is float:
        string = "{:E}".format(number)
        ls, rs = string.split('E')
        ls = ls.rstrip('0')  # remove the trailing zeros
        ls0, ls1 = ls.split('.')
        ls = ls0 + ls1
        power = int(rs) if len(rs) > 1 else 0
        if len(ls1) > 0:
            power += int(len(ls1))
        string = ls + identifier + str(power)
        return string
    if type(number) is int:
        return identifier + str(number)

def create_config_paths(exp_dir):
    files = os.listdir(exp_dir)
    files = [x for x in files if 'config' in x]
    n_config = (len(files) // 2) + 1

    csv_path = os.path.join(exp_dir, 'config%i.csv' % n_config)
    pk_path = os.path.join(exp_dir, 'config%i.pk' % n_config)

    return csv_path, pk_path


def save_config_csv_and_pickle(config, csv_path, pk_path):
    
    with open(csv_path, 'w') as f:
        for key, val in config.items():
            if not 'dir' in key:
                f.write("%s,%s\n" % (key, val))

    with open(pk_path, 'wb') as f:
        pk.dump(config, f)


def are_we_loading(load_it, load_dir):
    if load_it > 0 or len(load_dir) > 0:
        if len(load_dir) > 0:
            if load_it == 0:
                print('Provide load iteration load_it if loading')
                sys.exit()
            return True
    return False


def are_we_loading_pretraining(root, system, pretrain, pre_lr, n_pre_it, ansatz_hyperparameter_name):
    pretrain_dir = os.path.join(root, system, 'pretrained')
    hyperparameter_name = '%s_%s' % (n2n(pre_lr, 'lr'), n2n(n_pre_it, 'i'))
    pretrain_path = os.path.join(pretrain_dir, ansatz_hyperparameter_name + '_' + hyperparameter_name + '.pk')
    make_dir(pretrain_dir)
    if pretrain:
        return False, pretrain_path
    if n_pre_it == 0:
        return False, ''
    return True, pretrain_path


def to_array(object):
    if object is None: return None
    elif type(object) in (float, int, bool): return object
    else: return jnp.array(object)


def dict_entries_to_array(dictionary):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k] = to_array(v)
    return new_dict


def get_system(system, 
               n_el,
               density_parameter):

    PATH = os.path.abspath(os.path.dirname(__file__))
    systems_data = toml.load(os.path.join(PATH, 'systems_data.toml'))
    if not system in systems_data:
        pass
        sys.exit('system not in db')
    if system == 'HEG':
        systems_data = {'HEG':{
            **systems_data['HEG'],  # real_basis, pbc
            'r_atoms': None,  # enforces n_atoms = 1 in the molecule class HACKY
            'z_atoms': None,
            'n_el': n_el,
            'n_el_atoms': [n_el],
            'density_parameter': density_parameter}              
        }

    return dict_entries_to_array(systems_data[system])


def get_run(exp_dir):
    exps = os.listdir(exp_dir)
    nums = [int(e[-1]) for e in exps if 'run' in e]
    trial = 0
    while True:
        if trial not in nums:
            break
        trial += 1
    return trial


def get_n_devices():
    n_devices = len(jax.devices())
    return n_devices


def setup(system: str = 'Be',
          name = None,
          save_every: int = 1000,
          print_every: int = 100,

          r_atoms=None,
          z_atoms=None,
          n_el=None,
          n_el_atoms=None,
          ignore_toml=False,
          pbc=False,
          real_basis=None,
          density_parameter=None,
          real_cut=6,
          reciprocal_cut=6,
          kappa=0.5,

          opt: str = 'kfac',
          lr: float = 1e-4,
          damping: float = 1e-3,
          norm_constraint: float = 1e-4,
          n_it: int = 1000,
          n_walkers: int = 512,

          step_size: float = 0.02,
          correlation_length: int = 10,

          n_layers: int = 2,
          n_sh: int = 32,
          n_ph: int = 8,
          n_det: int = 2,
          scalar_inputs: bool = False,
          n_periodic_input: int = 3,
          orbitals: str = 'anisotropic',

          pre_lr: float = 1e-4,
          n_pre_it: int = 1000,
          pretrain: bool = False,
          load_pretrain: bool = False,

          load_it: int = 0,
          load_dir: str = '',

          distribute: bool = True, 
          debug: bool = False,

          seed: int = 369):

    n_devices = get_n_devices()
    assert n_walkers % n_devices == 0

    today = datetime.datetime.now().strftime("%d%m%y")
    version = today # version = subprocess.check_output(["git", "describe"]).strip()
    root = os.path.join(os.getcwd(), 'experiments')

    if name is None: name = 'junk'

    loading = are_we_loading(load_it, load_dir)
    ansatz_hyperparameter_name = '%s_%s_%s_%s' % (n2n(n_sh, 's'), n2n(n_ph, 'p'), n2n(n_layers, 'l'), n2n(n_det, 'det'))
    if loading: exp_dir = load_dir
    else:
        hyperparameter_name = '%s_%s_%s_%s_%s_' % (opt, n2n(lr, 'lr'), n2n(damping, 'd'), n2n(norm_constraint, 'nc'), n2n(n_walkers, 'm'))
        exp_dir = join_and_create(root, system, today, name, hyperparameter_name + ansatz_hyperparameter_name)
        run = 'run%i' % get_run(exp_dir)
        exp_dir = os.path.join(exp_dir, run)

    pretrain_dir = join_and_create(root, system, 'pretrained')
    hyperparameter_name = '%s_%s' % (n2n(pre_lr, 'lr'), n2n(n_pre_it, 'i'))
    pre_path = join_and_create(pretrain_dir, ansatz_hyperparameter_name + '_' + hyperparameter_name + '.pk')

    events_dir = join_and_create(exp_dir, 'events')
    timing_dir = join_and_create(events_dir, 'timing')
    models_dir = join_and_create(exp_dir, 'models')
    opt_state_dir = join_and_create(models_dir, 'opt_state')

    system_config = get_system(system,
                               n_el, 
                               density_parameter)

    csv_cfg_path, pk_cfg_path = create_config_paths(exp_dir)

    config = {'version': version,
              'seed': seed,
              'n_devices': n_devices,
              'save_every': save_every,
              'print_every': print_every,

              # PATHS
              'exp_dir': exp_dir,
              'events_dir': events_dir,
              'models_dir': models_dir,
              'opt_state_dir': opt_state_dir,
              'pre_path': pre_path,
              'timing_dir': timing_dir,
              'csv_cfg_path':csv_cfg_path,
              'pk_cfg_path': pk_cfg_path,

              # SYSTEM
              **system_config,
              'system': system,
              'real_cut': real_cut,
              'reciprocal_cut': reciprocal_cut,
              'kappa': kappa,

              # ANSATZ
              'n_layers': n_layers,
              'n_sh': n_sh,
              'n_ph': n_ph,
              'n_det': n_det,
              'scalar_inputs': scalar_inputs, 
              'n_periodic_input': n_periodic_input,
              'orbitals': orbitals, 

              # TRAINING HYPERPARAMETERS
              'opt': opt,
              'lr': lr,
              'damping': damping,
              'norm_constraint': norm_constraint,
              'n_it': n_it,
              'load_it': load_it,
              'n_walkers': n_walkers,
              'n_walkers_per_device': n_walkers // n_devices,
              'step_size': step_size,
              'correlation_length': correlation_length,

              # PRETRAINING HYPERPARAMETERS
              'pre_lr': pre_lr,
              'n_pre_it': n_pre_it,
              'load_pretrain': load_pretrain,
              'pretrain': pretrain

    }

    save_config_csv_and_pickle(config, csv_cfg_path, pk_cfg_path)

    for k, v in config.items():
        print(k, '\t\t', v)

    if distribute: os.environ['DISTRIBUTE'] = 'True'
    if debug: os.environ['DEBUG'] = 'True'

    return config


def write_summary_to_cfg(path, summary):
    with open(path, 'a') as f:
        f.write("# Final summary \n")
        for key, val in summary.items():
            f.write("%s,%s\n" % (key, val))


def load_config_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    new_config = {}
    for k, v in x.items():
        if type(v) == type(np.array([1.])):
            v = jnp.array(v)
        new_config[k] = v
        
    return new_config


def compare(a, b):
    print(jnp.isclose(a, b, atol=1e-6).all())
    print(jnp.max(jnp.abs(a - b)))
    return jnp.abs(a - b).sum()


def nans_in_tree(arg):
    arg, _ = tree_flatten(arg)
    print(jnp.array([jnp.isnan(x).any() for x in arg]).any())


def nans(arg):
    print(jnp.isnan(arg).any())


def walker_checks(mol, vwf, params, walkers, r_atoms):
    assert (walkers.dot(mol.inv_real_basis) > 0.).all()
    ae_vectors = walkers[:, None, ...] - r_atoms[None, None, ...]
    assert (ae_vectors.dot(mol.inv_real_basis) < 0.5).all()
    log_psi = vwf(params, walkers.squeeze(0))
    print('nans in log_psi ', jnp.isnan(log_psi).any())
    flat_params, map = tree_flatten(params)
    nans_in_params = jnp.array([jnp.isnan(x).any() for x in flat_params])
    print('nans in params ', nans_in_params.any())
    print('passed')


def compute_last10(e_means):
    n_it = len(e_means)
    if n_it < 10:
        return float(jnp.mean(jnp.array(e_means)))
    idx = n_it // 10
    return float(jnp.mean(jnp.array(e_means[-idx:])))


class Logging():
    def __init__(self,
                 events_dir,
                 models_dir,
                 opt_state_dir,
                 save_every,
                 print_every=None,
                 **kwargs):

        self.summary_writer = SummaryWriter(events_dir)  # comment= to add tag to file
        self.save_every = save_every
        self.print_every = print_every

        self.events_dir = events_dir
        self.opt_state_dir = opt_state_dir
        self.models_dir = models_dir

        self.times = {}
        self.e_means = []
        self.data = {}

        self.walkers = None

    def update_summary(self, name, value):
        value = np.array(value)
        if not np.isnan(value).any():
            self.summary[name] = value


    def writer(self, name, value, step):
        self.summary_writer.add_scalar(name, value, step)
        if self.data.get(name) is None:
            self.data[name] = [value]
        else:
            self.data[name].append(value)
        self.update_summary(name, value)

    def log(self,
            step,
            opt_state=None,
            params=None,
            e_locs=None,
            acceptance=None,
            walkers=None,
            **kwargs):

        self.timer(step, 'iteration')

        # for key, val in kwargs.items():
        #     if hasattr(val, "__len__"):
        #         self.write(step, key, val)

        self.printer = {}
        self.summary = {}

        if e_locs is not None:
            e_mean = float(jnp.mean(e_locs))
            e_std = float(jnp.std(e_locs))
            self.e_means.append(e_mean)
            self.writer('e_mean', e_mean, step)
            self.writer('e_std', e_std, step)
            e_mean_mean = compute_last10(self.e_means)
            self.writer('e_mean_mean', e_mean_mean, step)
            self.printer['e_mean'] = e_mean
            self.printer['e_std'] = e_std
            self.printer['e_mean_mean'] = e_mean_mean

        if acceptance is not None:
            self.writer('acceptance', float(acceptance), step)
            self.printer['acceptance'] = acceptance

        if self.times.get('iteration') is not None:
            self.printer['t_per_it'] = self.times['iteration'][-1][1] \
                                       / (self.times['iteration'][-1][0] - self.times['step0'] + 1.)

        if not self.print_every == 0:
            if step % self.print_every == 0:
                self.print_pretty(step)

        if step % self.save_every == 0:
            self.save_state(step, opt_state=opt_state, params=params, walkers=walkers)
            save_pk(self.data, os.path.join(self.events_dir, 'data.pk'))

        self.params = params

    # def save_data(self):
        # save_pk(self.data, os.path.join(self.events_dir, 'data.pk'))

    def save_state(self, step, opt_state=None, params=None, walkers=None):
        if opt_state is not None:
            try:  # differences between kfac and adam state objects
                save_pk(opt_state, os.path.join(self.opt_state_dir, 'i%i.pk' % step))
            except TypeError as e:
                opt_state = optimizers.unpack_optimizer_state(opt_state)
                save_pk(opt_state, os.path.join(self.opt_state_dir, 'i%i.pk' % step))
                # best_opt_state = optimizers.pack_optimizer_state(best_params)

        if params is not None:
            save_pk(params, os.path.join(self.models_dir, 'i%i.pk' % step))

        if walkers is not None:
            save_pk(walkers, os.path.join(self.models_dir, 'i%i_walkers.pk' % step))

        if not len(self.times) == 0:
            save_pk(self.times, os.path.join(self.events_dir, 'timings.pk'))

    def timer(self, step, key):
        if key not in self.times:
            if not step < 3: 
                t0 = time.time()
                self.times[key] = [[step, t0]]
                self.times['t0'] = t0
                self.times['step0'] = step
        else:
            self.times[key].append([step, time.time() - self.times['t0']])

    def print_pretty(self, step):
        string = 'step %i |' % step
        for k, val in self.printer.items():
            string += ' %s %.4f |' % (k, val)
        print(string)



def results_name(root='.'):
    files = os.listdir(root)
    files = [f for f in files if 'results' in f]
    number = 0
    if len(files) == 0:
        return 'results%i.csv' % number
    number_in_files = True
    while number_in_files:
        for f in files:
            if not str(number) in f:
                number_in_files = False
                break
        if number_in_files:
            number += 1
    return 'results%i.csv' % number



def write_into_common_csv_or_dump(row, root='.'):
    name = results_name(root)
    try:
        with open(name, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except:
        with open('results_%i.csv' % row[0], 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def check_if_nan(tensor, desc):
    if jnp.isnan(tensor).any():
        # save_pk(tensor, 'nan_%s.pk' % desc)
        return True
    return False


def capture_nan(tensor, desc, stop_prev):
    if isinstance(tensor, jnp.ndarray):
        stop = check_if_nan(tensor, desc)
    else:
        tree, _ = tree_flatten(tensor)
        stop = False
        for tensor in tree: 
            tmp_stop = check_if_nan(tensor, desc)
            stop = stop or tmp_stop
    if stop:
        print('nans in %s' % desc)
    return stop or stop_prev


if __name__ == '__main__':
    # config = setup(system='LiH')
    c = load_pk()