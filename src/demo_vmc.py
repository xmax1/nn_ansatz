
from nn_ansatz import setup, run_vmc, compare_einsum, run_vmc_debug, check_inf_nan, create_jastrow_factor
import jax.numpy as jnp


cfg = setup(system='HEG',
                    pretrain=False,
                    n_walkers=512,
                    n_layers=2,
                    n_sh=64,
                    step_size=0.05,
                    n_ph=16,
                    n_el=7,
                    orbitals='real_plane_waves',
                    simulation_cell=(1, 1, 1),
                    density_parameter=1., 
                    opt='kfac',
                    n_det=1,
                    print_every=10,
                    save_every=2500,
                    input_activation_nonlinearity='cos+19kpoints',
                    lr=1e-3,
                    n_it=10000,
                    name='junk')

log = run_vmc(cfg)

# walkers, grads, pe, ke, probs = run_vmc_debug(cfg)

# walkers_check = check_inf_nan(walkers)
# pe_check = check_inf_nan(pe)
# ke_check = check_inf_nan(ke)
# prob_check = check_inf_nan(probs)

# if pe_check:
#     print('pe')
#     test = pe

# if ke_check:
#     print('ke')
#     test = ke

# if prob_check:
#     print('prob')
#     test = probs

# idxs = jnp.where(jnp.isnan(test))[0]
# print(walkers[:, idxs])

# jastrow = create_jastrow_factor(7, 7, 1., )

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
