
import sys
sys.path.append('..')
from nn_ansatz import approximate_pair_distribution_function, load_pk

import numpy as np

cfg = load_pk('/home/amawi/projects/nn_ansatz/src/exp/experiments/HEG/051121/adam_vs_kfac/kfac_1lr-4_1d-3_1nc-4_m512_s64_p32_l2_det1/run0/config1.pk')
pdf_saverio = np.loadtxt('/home/amawi/projects/nn_ansatz/src/exp/gofr.txt')[:, 0]


pdf, pdf_std = approximate_pair_distribution_function(cfg, load_it=50000, n_bins=len(pdf_saverio)-1, n_it=10, walkers=None)

# pdf_e1, pdf_std_e1 = np.squeeze(pdf[:, 0, 1]), np.squeeze(pdf_std[:, 0, 1])
save_me = np.stack([pdf, pdf_std, pdf_saverio], axis=-1)
np.savetxt('/home/amawi/projects/nn_ansatz/src/exp/gofr_max.txt', save_me)

print(pdf)
print(pdf_saverio)

assert len(pdf) == len(pdf_saverio)
