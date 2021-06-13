
import datetime
import os
import pickle as pk
from torch.utils.tensorboard import SummaryWriter
import time
from jax import jit
from jax.experimental import optimizers
import jax
import jax.random as rnd
import jax.numpy as jnp
import toml
PATH = os.path.abspath(os.path.dirname(__file__))
systems_data = toml.load(os.path.join(PATH, 'systems_data.toml'))


def split_variables_for_pmap(n_devices, *args):
    
    new_args = []
    for arg in args:
        if type(arg) in (float, int):
            new_arg = jnp.array(arg).repeat(n_devices)
        new_args.append(new_arg)
    
    # not sure how to unpack the list if there is just one element
    if len(new_args) == 1:
        return new_args[0]
    return new_args

@jit
def key_gen(keys):
    """
    keys: (n_devices, 2)
    Pseudocode:
        - generate the new keys for each device and put into a new array
        - split the array along axis 1 so there are 2 arrays (keys and subkeys)
        - squeeze the middle axis out
    """
    keys = jnp.array([rnd.split(key) for key in keys])
    keys = jnp.split(keys, 2, axis=1)
    return [x.squeeze(axis=1) for x in keys]

# key_gen = lambda keys: [x.squeeze() for x in jnp.array([rnd.split(key) for key in keys]).split(2, axis=1)]


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x


def compare(tc_arr, jnp_arr):
    arr = tc_arr.detach().cpu().numpy()
    diff = arr - jnp_arr
    print('l1-norm: ', jnp.mean(jnp.abs(diff)))


def remove_aux(fn, axis=0):
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


def save_config_csv_and_pickle(config, exp_dir):
    files = os.listdir(exp_dir)
    files = [x for x in files if 'config' in x]
    n_config = (len(files) // 2) + 1

    csv_path = os.path.join(exp_dir, 'config%i.csv' % n_config)
    with open(csv_path, 'w') as f:
        for key, val in config.items():
            if not 'dir' in key:
                f.write("%s,%s\n" % (key, val))

    pk_path = os.path.join(exp_dir, 'config%i.pk' % n_config)
    with open(pk_path, 'wb') as f:
        pk.dump(config, f)


def are_we_loading(load_it, load_dir):
    if load_it > 0 or len(load_dir) > 0:
        if len(load_dir) > 0:
            if load_it == 0:
                print('Provide load iteration load_it if loading')
                exit()
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


def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def array_if_not_none(object):
    return jnp.array(object) if object is not None else None


def get_system(system, r_atoms, z_atoms, n_el, n_el_atoms, 
       periodic_boundaries, real_basis, unit_cell_length, real_cut, reciprocal_cut, kappa, ignore_toml):
    
    if ignore_toml or not system in systems_data:
        if n_el is None and n_el_atoms is None:
            if system == 'Be':
                n_el = 4
            else:
                exit('Need number of electrons as minimum input or system in toml file')
        print('Assuming atomic system')
        n_el_atoms = jnp.array([n_el]) if n_el is not None else n_el_atoms
        n_el = n_el if n_el is not None else int(jnp.sum(n_el_atoms))
        if r_atoms is None:
            r_atoms = jnp.array([[0.0, 0.0, 0.0]])
        if z_atoms is None:
            z_atoms = jnp.ones((1,)) * n_el
        return r_atoms, z_atoms, n_el, n_el_atoms, \
        {'periodic_boundaries': periodic_boundaries, 
        'real_basis': real_basis, 
        'unit_cell_length': unit_cell_length,
        'real_cut': real_cut,
        'reciprocal_cut': reciprocal_cut,
        'kappa': kappa}
    else:
        
        d = systems_data[system]
        return jnp.array(d['r_atoms']), \
               jnp.array(d['z_atoms']), \
               d['n_el'], \
               jnp.array(d['n_el_atoms']), \
               {'periodic_boundaries': d.get('periodic_boundaries', periodic_boundaries), 
                'real_basis': array_if_not_none(d.get('real_basis', real_basis)),
                'unit_cell_length': d.get('unit_cell_length', unit_cell_length),
                'real_cut': d.get('real_cut', real_cut),
                'reciprocal_cut': d.get('reciprocal_cut', reciprocal_cut),
                'kappa': d.get('kappa', kappa)}


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
          name: str = '',
          exp: bool = False,
          save_every: int = 1000,
          print_every: int = 0,

          r_atoms=None,
          z_atoms=None,
          n_el=None,
          n_el_atoms=None,
          ignore_toml=False,
          periodic_boundaries=False,
          real_basis=None,
          unit_cell_length=None,
          real_cut=5,
          reciprocal_cut=5,
          kappa=1,

          opt: str = 'kfac',
          lr: float = 1e-4,
          damping: float = 1e-4,
          norm_constraint: float = 1e-4,
          n_it: int = 1000,
          n_walkers: int = 1024,

          step_size: float = 0.02,
          correlation_length: int = 10,

          n_layers: int = 2,
          n_sh: int = 32,
          n_ph: int = 8,
          n_det: int = 2,

          pre_lr: float = 1e-4,
          n_pre_it: int = 1000,
          pretrain: bool = False,
          load_pretrain: bool = False,

          load_it: int = 0,
          load_dir: str = '',

          seed: int = 369):

    n_devices = get_n_devices()
    assert n_walkers % n_devices == 0

    today = datetime.datetime.now().strftime("%d%m%y")

    # version = subprocess.check_output(["git", "describe"]).strip()
    version = today

    # root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'experiments')
    root = os.path.join(os.getcwd(), 'experiments')

    if len(name) == 0:
        name = today

    if not exp:
        name = 'junk'

    loading = are_we_loading(load_it, load_dir)
    ansatz_hyperparameter_name = '%s_%s_%s_%s' % (n2n(n_sh, 's'), n2n(n_ph, 'p'), n2n(n_layers, 'l'), n2n(n_det, 'det'))
    if loading:
        exp_dir = load_dir
    else:
        hyperparameter_name = '%s_%s_%s_%s_%s_' % (opt, n2n(lr, 'lr'), n2n(damping, 'd'), n2n(norm_constraint, 'nc'), n2n(n_walkers, 'm'))
        exp_dir = os.path.join(root, system, name, hyperparameter_name + ansatz_hyperparameter_name)
        make_dir(exp_dir)
        run = 'run%i' % get_run(exp_dir)
        exp_dir = os.path.join(exp_dir, run)

    pretrain_dir = os.path.join(root, system, 'pretrained')
    hyperparameter_name = '%s_%s' % (n2n(pre_lr, 'lr'), n2n(n_pre_it, 'i'))
    pre_path = os.path.join(pretrain_dir, ansatz_hyperparameter_name + '_' + hyperparameter_name + '.pk')
    make_dir(pre_path)

    events_dir = os.path.join(exp_dir, 'events')
    timing_dir = os.path.join(events_dir, 'timing')
    models_dir = os.path.join(exp_dir, 'models')
    opt_state_dir = os.path.join(models_dir, 'opt_state')
    [make_dir(x) for x in [events_dir, models_dir, opt_state_dir]]

    r_atoms, z_atoms, n_el, n_el_atoms, cell_config \
        = get_system(system, r_atoms, z_atoms, n_el, n_el_atoms, 
        periodic_boundaries, real_basis, unit_cell_length, real_cut, reciprocal_cut, kappa, ignore_toml)

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

              # SYSTEM
              'system': system,
              'r_atoms': r_atoms,
              'z_atoms': z_atoms,
              'n_el': n_el,
              'n_el_atoms': n_el_atoms,
              **cell_config,

              # ANSATZ
              'n_layers': n_layers,
              'n_sh': n_sh,
              'n_ph': n_ph,
              'n_det': n_det,

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

    save_config_csv_and_pickle(config, exp_dir)

    for k, v in config.items():
        print(k, '\t\t', v)

    return config


def save_pk(data, path):
    with open(path, 'wb') as f:
        pk.dump(data, f)


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

        self.writer = SummaryWriter(events_dir)
        self.save_every = save_every
        self.print_every = print_every

        self.events_dir = events_dir
        self.opt_state_dir = opt_state_dir
        self.models_dir = models_dir

        self.times = {}
        self.e_means = []

    def log(self,
            step,
            opt_state=None,
            params=None,
            e_locs=None,
            acceptance=None,
            **kwargs):

        self.timer(step, 'iteration')

        # for key, val in kwargs.items():
        #     if hasattr(val, "__len__"):
        #         self.write(step, key, val)

        self.printer = {}

        if e_locs is not None:
            e_mean = float(jnp.mean(e_locs))
            e_std = float(jnp.std(e_locs))
            self.e_means.append(e_mean)
            self.writer.add_scalar('e_mean', e_mean, step)
            self.writer.add_scalar('e_std', e_std, step)
            e_mean_mean = compute_last10(self.e_means)
            self.writer.add_scalar('e_mean_mean', e_mean_mean, step)
            self.printer['e_mean'] = e_mean
            self.printer['e_std'] = e_std
            self.printer['e_mean_mean'] = e_mean_mean

        if acceptance is not None:
            self.writer.add_scalar('acceptance', float(acceptance), step)
            self.printer['acceptance'] = acceptance

        if self.times.get('iteration') is not None:
            self.printer['t_per_it'] = self.times['iteration'][-1][1] \
                                       / (self.times['iteration'][-1][0] - self.times['step0'] + 1.)

        if not self.print_every == 0:
            if step % self.print_every == 0:
                self.print_pretty(step)

        if step % self.save_every == 0:
            self.save_state(step, opt_state=opt_state, params=params)

    def save_state(self, step, opt_state=None, params=None):
        if opt_state is not None:
            try:  # differences between kfac and adam state objects
                save_pk(opt_state, os.path.join(self.opt_state_dir, 'i%i.pk' % step))
            except TypeError as e:
                opt_state = optimizers.unpack_optimizer_state(opt_state)
                save_pk(opt_state, os.path.join(self.opt_state_dir, 'i%i.pk' % step))
                # best_opt_state = optimizers.pack_optimizer_state(best_params)

        if params is not None:
            save_pk(params, os.path.join(self.models_dir, 'i%i.pk' % step))

        if not len(self.times) == 0:
            save_pk(self.times, os.path.join(self.events_dir, 'timings.pk'))

    def timer(self, step, key):
        if key not in self.times:
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




if __name__ == '__main__':
    config = setup(system='LiH')