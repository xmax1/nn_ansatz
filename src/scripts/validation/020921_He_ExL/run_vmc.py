import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')

from nn_ansatz import *

# factor = float(sys.argv[1])

# config = setup(system='LiSolidBCC',
#                opt='adam',
#                n_it=500,
#                name='%.2fL' % factor,
#                n_walkers=128,
#                save_every=1000)

# config['unit_cell_length'] *= factor

# log = run_vmc(config)

# e_mean_mean = log.data['e_mean_mean'][-1]

# row = [factor, e_mean_mean]

# with open('LiSolidBCC_extrapolation.csv', 'a+') as f:
#     writer = csv.writer(f)
#     writer.writerow(row)

# write_into_common_csv_or_dump(row)

# single run case
# config = setup(system='LiSolidBCC',
#                opt='adam',
#                n_it=10000,
#                name='solid',
#                n_walkers=512,
#                save_every=1000)

# log = run_vmc(config)
# e_mean_mean = log.data['e_mean_mean'][-1]
# print(e_mean_mean)

# isolated case
config = setup(system='He',
               opt='kfac',
               n_it=200,
               name='isolated',
               n_walkers=128,
               save_every=1000)

log = run_vmc(config)
e_mean_mean = log.data['e_mean_mean'][-1]
print(e_mean_mean)





