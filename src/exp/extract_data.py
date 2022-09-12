

import sys
sys.path.append('/home/energy/amawi/projects/nn_ansatz/src')
import os
os.environ['CUDA_VISIBLE_DEVICES'] =''

from nn_ansatz.plot import get_data
import pandas as pd
pd.set_option("display.precision", 8)


name = '/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/final0301'
x = 'density_parameter'
y = 'equilibrated_energy_mean'
yerr = 'equilibrated_energy_sem'
title = 'denisty parameter sweep'
group = True

# data = get_data(name, groupby=x, dicts=[])
data = get_data(name, dicts=[])

xs = data[x]
ys = data[y]
yerrs = data[yerr]

# graph = plot_scatter(xs, 
#                      ys, 
#                      yerrs=yerrs,
#                      title=title) 
# show(graph)

print(data[['n_el', x, y, yerr, 'energy (Ry)']].sort_values(by=['n_el', x]).style.hide_index())