
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.85'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nn_ansatz import *
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange
from jax import pmap, vmap, grad

cfg = setup(system='LiSolidBCC',
                    n_pre_it=0,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=32,
                    step_size=0.02,
                    n_ph=16,
                    scalar_inputs=False,
                    orbitals='anisotropic',
                    n_periodic_input=1,
                    opt='kfac',
                    n_det=4,
                    print_every=100,
                    save_every=2500,
                    lr=1e-4,
                    n_it=10000,
                    name=None)
log = run_vmc(cfg)
