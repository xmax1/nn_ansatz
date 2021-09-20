#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')


# In[14]:


import jax
from jax import pmap, vmap
from jax.tree_util import tree_flatten
from jax.experimental.optimizers import adam
import jax.numpy as jnp
from jax import lax

import numpy as np
from tqdm.notebook import trange
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import cm

from nn_ansatz import *


# In[16]:


cfg = config = setup(system='LiSolidBCC',
               n_pre_it=0,
               n_walkers=64,
               n_layers=2,
               n_sh=32,
               n_ph=8,
               opt='kfac',
               n_det=2,
               print_every=1,
               save_every=5000,
               n_it=1000)

logger = Logging(**cfg)

keys = rnd.PRNGKey(cfg['seed'])
if bool(os.environ.get('DISTRIBUTE')) is True:
    keys = rnd.split(keys, cfg['n_devices']).reshape(cfg['n_devices'], 2)

mol = SystemAnsatz(**cfg)

pwf = pmap(create_wf(mol), in_axes=(None, 0))
vwf = create_wf(mol)
jswf = jit(create_wf(mol, signed=True))
compute_ae_vectors = jit(partial(compute_ae_vectors_periodic_i, unit_cell_length=mol.unit_cell_length))

sampler = create_sampler(mol, vwf)

# params = initialise_params(mol, keys)
params = load_pk('params.pk')
# walkers = initialise_walkers(mol, vwf, sampler, params, keys, walkers=None)
# save_pk(walkers, 'walkers_no_infs.pk')
walkers = load_pk('walkers_no_infs.pk')

ke = jit(create_local_kinetic_energy(vwf))
pe = jit(create_potential_energy(mol))
keep_in_boundary = jit(keep_in_boundary)


# In[20]:


@jit
def spline(walker, r_atoms, unit_cell_length):
    ae_vector = compute_ae_vectors(walker, r_atoms)
    ae_vector = jnp.where(ae_vector < -0.25 * unit_cell_length, -unit_cell_length**2/(8.*(unit_cell_length + 2.*ae_vector)), ae_vector)
    ae_vector = jnp.where(ae_vector > 0.25 * unit_cell_length, unit_cell_length**2/(8.*(unit_cell_length - 2.*ae_vector)), ae_vector)
    return ae_vector

spline = vmap(spline, in_axes=(0, None, None))


# In[39]:


# move an electron in a line over one of the nuclei with another electron in its path
# jwalker.shape (1, 6, 3)
# r_atoms.shape (2, 3)
walker = np.array(walkers[:, 0, ...])
rx0, ry0, rz0 = np.array(mol.r_atoms[1, :])
walker[0, 0, -1] = rz0
walker[0, 1, -1] = rz0
walker[0, 3, -1] = rz0

n_points = 1000
X = np.linspace(rx0-mol.unit_cell_length, rx0+mol.unit_cell_length, n_points)
# idx = np.argmin(np.abs(X))
# X[idx] = 0.0

print(rz0)

data = []
for i, y1 in enumerate(X):
    for x1 in X:
        walker[0, 0] = jnp.array([[x1, y1, rz0]])
        jwalker = jnp.array(walker)

        # print(jwalker.shape, mol.r_atoms.shape)

        ae_vectors = spline(jwalker, mol.r_atoms, mol.unit_cell_length)
        mask = jnp.isinf(ae_vectors) 
        max_ninf = mask.sum(-1).max()

        jwalker = keep_in_boundary(jwalker, mol.real_basis, mol.inv_real_basis)
        log_psi, sign = jswf(params, jwalker)
        amplitude = sign * jnp.exp(log_psi)
        probability = amplitude**2
        
        potential_energy = pe(jwalker, mol.r_atoms, mol.z_atoms)
        kinetic_energy = ke(params, jwalker)
        local_energy = potential_energy + kinetic_energy
        tmp = [y1, x1, amplitude.mean(), probability.mean(), local_energy.mean(), potential_energy.mean(), kinetic_energy.mean(), max_ninf]
        data.append(tmp)
    print('row %i' % (i+1), ' / %i' % n_points)

data = np.array(data)
data = pd.DataFrame(data, columns=['y', 'x', 'amps', 'probs', 'local_e', 'potential_e', 'kinetic_e', 'max_ninf'])
save_pk(data, 'electron_sweep.pk')
print(data)


# In[38]:


XX, YY = np.meshgrid(X, X)

fig, axs = plt.subplots(3, 2, figsize=(14,14))
axs = [x for y in axs for x in y]

names = ['amps', 'local_e', 'potential_e', 'kinetic_e', 'max_ninf']

for ax, name in zip(axs, names):
    plot_data = data.pivot(index='x', columns='y', values=name).values
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
    
    cmap = cm.get_cmap('cool', 3) if name is 'max_ninf' else cm.cool
    if name is not 'max_ninf':
        vmin, vmax = plot_data.min().min(), plot_data.max().max()  
    else:
        vmin, vmax = 0.5, 3.5
    
    plot_data = data.pivot(index='x', columns='y', values=name).values
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
    p = ax.pcolor(XX, YY, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.scatter(mol.r_atoms[:, 0], mol.r_atoms[:, 1], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(name)
    cb = fig.colorbar(p, ax=ax)
    # fig.suptitle('k = %.2f, min energy = %.6f / max energy  = %.6f' % (kappa, plot_data.min().min(), plot_data.max().max()))
fig.tight_layout()
plt.savefig('electron_sweep.png', facecolor='white', transparent=False)
plt.show()


# In[17]:


plot_data = data.pivot(index='x', columns='y', values='amps')
print(plot_data)


# In[ ]:


# def plot_refs(ax, refs, lim):
#     yline = [y for y in range(-lim, lim)]
#     for name, loc, color in refs:
#         ax.plot([loc for _ in yline], yline, color=color, label=name)

# print(walker)
# lim = 20.
# fig, axs = plt.subplots(2, 2, figsize=(15, 8))
# axs = [x for y in axs for x in y]
# names = ['amps', 'local_e', 'potential_e', 'kinetic_e']
# X = data['x']

# refs = [['nuclei', 0.0, 'g'],
#         ['spline_boundary1', -mol.unit_cell_length / 4., 'b'],
#         ['spline_boundary2', mol.unit_cell_length / 4., 'b'],
#         ['min_image_boundary_half_cell1', -mol.unit_cell_length / 2., 'r'],
#         ['min_image_boundary_half_cell1', mol.unit_cell_length / 2., 'r']]


# for ax, name in zip(axs, names):
#     y = data[name]
#     ax.plot(X, y)
#     y = np.clip(y, -lim, lim)
#     ymax, ymin = np.nanmax(y), np.nanmin(y)
#     ax.set_ylabel(name)
#     ax.set_xlabel('x_displacement')
#     ax.set_ylim(ymin, ymax)

#     plot_refs(ax, refs)

# axs[-1].legend(bbox_to_anchor=(1,1), loc="upper left")
# fig.tight_layout()


# In[ ]:


print(data)


# In[ ]:




