
import datetime
import os
import pickle as pk
from torch.utils.tensorboard import SummaryWriter
import time
from jax.experimental import optimizers
import jax.numpy as jnp
import toml
PATH = os.path.abspath(os.path.dirname(__file__))
systems_data = toml.load(os.path.join(PATH, 'systems_data.toml'))


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


def are_we_pretraining(root, system, pretrain, pre_lr, n_pre_it, ansatz_hyperparameter_name):
    pretrain_dir = os.path.join(root, system, 'pretrained')
    hyperparameter_name = '%s_%s' % (n2n(pre_lr, 'lr'), n2n(n_pre_it, 'i'))
    pretrain_path = os.path.join(pretrain_dir, ansatz_hyperparameter_name + '_' + hyperparameter_name + '.pk')
    make_dir(pretrain_dir)
    if pretrain or not os.path.exists(pretrain_path):
        return False, pretrain_path
    return True, pretrain_path


def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def get_system(system, r_atoms, z_atoms, n_el, n_el_atoms):
    if system in systems_data:
        d = systems_data[system]
        return jnp.array(d['r_atoms']), jnp.array(d['z_atoms']), d['n_el'], jnp.array(d['n_el_atoms'])
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
    return r_atoms, z_atoms, n_el, n_el_atoms


def setup(system: str = 'Be',
          name: str = '',
          exp: bool = False,
          save_every: int = 1000,
          print_every: int = 0,

          r_atoms=None,
          z_atoms=None,
          n_el=None,
          n_el_atoms=None,

          opt: str = 'kfac',
          lr: float = 1e-4,
          damping: float = 1e-4,
          norm_constraint: float = 1e-4,
          n_it: int = 1000,
          n_walkers: int = 1024,
          step_size: float = 0.02,

          n_layers: int = 2,
          n_sh: int = 64,
          n_ph: int = 16,
          n_det: int = 2,

          pre_lr: float = 1e-4,
          n_pre_it: int = 1000,
          pretrain: bool = False,

          load_it: int = 0,
          load_dir: str = '',

          seed: int = 369):
    today = datetime.datetime.now().strftime("%d%m%y")

    # version = subprocess.check_output(["git", "describe"]).strip()
    version = today

    # root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'experiments')
    root = os.path.join(os.getcwd(), 'experiments')

    if not exp:
        root = os.path.join(root, 'junk')

    if len(name) == 0:
        name = today

    loading = are_we_loading(load_it, load_dir)
    ansatz_hyperparameter_name = '%s_%s_%s_%s' % (n2n(n_sh, 's'), n2n(n_ph, 'p'), n2n(n_layers, 'l'), n2n(n_det, 'det'))
    if loading:
        exp_dir = load_dir
    else:
        hyperparameter_name = '%s_%s_%s_%s_' % (opt, n2n(lr, 'lr'), n2n(damping, 'd'), n2n(norm_constraint, 'nc'))
        exp_dir = os.path.join(root, system, name, hyperparameter_name + ansatz_hyperparameter_name)
        make_dir(exp_dir)
        run = 'run%i' % len(os.listdir(exp_dir))
        exp_dir = os.path.join(exp_dir, run)

    load_pretrain, pre_path = are_we_pretraining(root, system, pretrain, pre_lr, n_pre_it, ansatz_hyperparameter_name)

    events_dir = os.path.join(exp_dir, 'events')
    timing_dir = os.path.join(events_dir, 'timing')
    models_dir = os.path.join(exp_dir, 'models')
    opt_state_dir = os.path.join(models_dir, 'opt_state')
    [make_dir(x) for x in [events_dir, models_dir, opt_state_dir]]

    r_atoms, z_atoms, n_el, n_el_atoms = get_system(system, r_atoms, z_atoms, n_el, n_el_atoms)

    config = {'version': version,
              'seed': seed,
              'save_every': save_every,
              'print_every': print_every,

              # PATHS
              'exp_dir': exp_dir,
              'events_dir': events_dir,
              'models_dir': models_dir,
              'opt_state_dir': opt_state_dir,
              'pre_path': pre_path,
              'timing_dir': timing_dir,

              # SYSTEM & ANSATZ
              'system': system,
              'r_atoms': r_atoms,
              'z_atoms': z_atoms,
              'n_el': n_el,
              'n_el_atoms': n_el_atoms,
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
              'step_size': step_size,

              # PRETRAINING HYPERPARAMETERS
              'pre_lr': pre_lr,
              'n_pre_it': n_pre_it,
              'load_pretrain': load_pretrain,

    }

    save_config_csv_and_pickle(config, exp_dir)

    for k, v in config.items():
        print(k, '\t\t', v)

    return config


def save_pk(data, path):
    with open(path, 'wb') as f:
        pk.dump(data, f)


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
            self.writer.add_scalar('e_mean', e_mean, step)
            self.writer.add_scalar('e_std', e_std, step)
            self.printer['e_mean'] = e_mean
            self.printer['e_std'] = e_std

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
            string += ' %s %.2f |' % (k, val)
        print(string)




if __name__ == '__main__':
    config = setup(system='LiH')