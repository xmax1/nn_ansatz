import sys
sys.path.append('..')
from nn_ansatz import setup, run_vmc, compare_einsum, approximate_energy, load_pk

n_it = 10000
cfg = setup(system='HEG',
            n_walkers=512,
            n_layers=2,
            n_sh=64,
            step_size=0.05,
            n_ph=32,
            orbitals='real_plane_waves',
            n_el = 7,
            input_activation_nonlinearity='sin+cos+bowl',
            n_periodic_input=1,
            opt='kfac',
            n_det=1,
            density_parameter=1.,
            lr = 1e-4,
            n_it=n_it,
            name='101121/heg_test')
log = run_vmc(cfg)
# cfg = load_pk('/home/amawi/projects/nn_ansatz/src/exp/experiments/HEG/101121/heg_test/kfac_1lr-4_1d-3_1nc-4_m512_s64_p32_l2_det4/run1/config1.pk')
approximate_energy(cfg, load_it=n_it)

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
