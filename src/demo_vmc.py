
from nn_ansatz import setup, run_vmc, compare_einsum, run_vmc_debug


cfg = setup(system='HEG',
                    n_pre_it=1000,
                    pretrain=True,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=32,
                    step_size=0.05,
                    n_ph=16,
                    n_el=7,
                    orbitals='real_plane_waves',
                    simulation_cell=(1, 1, 1),
                    density_parameter=1., 
                    n_periodic_input=1,
                    opt='kfac',
                    n_det=1,
                    print_every=50,
                    save_every=2500,
                    lr=1e-3,
                    n_it=1000,
                    name='pre_test')

walkers, grads, pe, ke = run_vmc(cfg)


# cfg = setup(system='HEG',
#                     n_pre_it=0,
#                     n_walkers=512,
#                     n_layers=2,
#                     n_sh=64,
#                     step_size=0.05,
#                     n_ph=32,
#                     scalar_inputs=False,
#                     orbitals='real_plane_waves',
#                     n_el = 7,
#                     n_periodic_input=1,
#                     opt='kfac',
#                     n_det = 1,
#                     density_parameter = 1.,
#                     lr = 1e-4,
#                     print_every=50,
#                     save_every=2500,
#                     n_it=30000)
# log = run_vmc(cfg)

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
