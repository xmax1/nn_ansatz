#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load_ext autoreload
# %autoreload 2
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.85'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
#jupyter nbconvert --to notebook --execute demo_notebook.ipynb --output demo_notebook.ipynb
# 


# In[3]:


import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')
from nn_ansatz import *
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange
from jax import pmap, vmap, grad


# In[13]:


# using routines
translations = [np.random.uniform(0, 0.5, (1, 3)) for i in range(6)]

e_means = []
for translation in translations:
    cfg = setup(system='LiSolidBCC',
                n_pre_it=0,
                n_walkers=1024,
                n_layers=3,
                n_sh=64,
                step_size=0.02,
                n_ph=16,
                scalar_inputs=False,
                orbitals='anisotropic',
                n_periodic_input=1,
                opt='adam',
                n_det=4,
                print_every=100,
                save_every=2500,
                lr=1e-4,
                n_it=25000)
    cfg['r_atoms'] += jnp.array(translation)
    log = run_vmc(cfg)
    e_mean = log.summary['e_mean_mean']
    e_means.append(e_mean)

with open('data.pk', 'wb') as f:
    pk.dump(e_means, f)


# In[12]:


import matplotlib.pyplot as plt
x = range(len(e_means))
fig, ax = plot.pretty_base(xlabel='experiment', ylabel='converged energy', 
                            title='convergence of vmc over translations of system nuclei Li$_2$',
                            ylines=[-7.5])
ax.scatter(x, [x/2. for x in e_means])
plt.savefig('system_translations.png')

