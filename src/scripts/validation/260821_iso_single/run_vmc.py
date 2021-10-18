import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')

import csv

from nn_ansatz import *

# factor = float(sys.argv[1])

config = setup(system='LiSolidBCC',
               orbitals='isotropic',
               scalar_inputs=True,
               opt='kfac',
               n_it=1000,
               n_walkers=128)

# config = setup(system='LiH',
#                orbitals='anisotropic',
#                opt='kfac',
#                n_it=1000,
#                n_walkers=128)

log = run_vmc(config)

e_mean_mean = log.data['e_mean_mean'][-1]

row = ['isotropic', e_mean_mean]

write_into_common_csv_or_dump(row)



