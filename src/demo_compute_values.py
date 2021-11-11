


from jax._src.dtypes import dtype
from nn_ansatz import setup, run_vmc, compare_einsum
from nn_ansatz import load_pk, approximate_energy
import pandas as pd
import os

exp_path = '/home/amawi/projects/nn_ansatz/src/experiments/HEG/041121/junk/adam_1lr-4_1d-3_1nc-4_m512_s64_p32_l2_det1/run2/config1.pk'
cfg = load_pk(exp_path)
cfg['load_it'] = 30000
values = approximate_energy(cfg, n_it=1000)
df = pd.DataFrame.from_dict(values, orient='columns')  # orient or index
print(df)
print(df.mean())
df.to_csv(os.path.join(cfg['events_dir'], 'approx_energy.csv'))



# cfg = setup(system='LiSolidBCC',
#                 n_pre_it=0,
#                 n_walkeroldtransform
#                 n_layers=2,
#                 n_sh=64,
#                 step_size=0.05,
#                 n_ph=32,
#                 scalar_inputs=False,
#                 orbitals='anisotropic',
#                 einsum=False,
#                 n_periodic_input=1,
#                 opt='adam',
#                 n_det=4,
#                 print_every=50,
#                 save_every=2500,
#                 lr=1e-4,
#                 n_it=30000)

# compare_einsum(cfg)
