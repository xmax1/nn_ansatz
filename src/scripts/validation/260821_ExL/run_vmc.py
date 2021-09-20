import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')

import csv

from nn_ansatz import *

factor = float(sys.argv[1])

config = setup(system='LiSolidBCC',
               opt='kfac',
               n_it=1000,
               name='%.2fL' % factor,
               n_walkers=128)

config['unit_cell_length'] *= factor

log = run_vmc(config)

e_mean_mean = log.data['e_mean_mean'][-1]

row = [factor, e_mean_mean]

write_into_common_csv_or_dump(row)



