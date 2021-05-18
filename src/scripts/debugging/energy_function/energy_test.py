import sys
sys.path.append('/home/xmax/projects/nn_ansatz/src')

from nn_ansatz import *

from jax import lax

def local_kinetic_energy_i(wf):
    """
    FUNCTION SLIGHTLY ADAPTED FROM DEEPMIND JAX FERMINET IMPLEMTATION
    https://github.com/deepmind/ferminet/tree/jax

    """
    def _lapl_over_f(params, walkers, d0s):
        walkers = walkers.reshape(-1)
        n = walkers.shape[0]
        eye = jnp.eye(n, dtype=walkers.dtype)
        grad_f = jax.grad(wf, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y, d0s)  # ensuring the input can be just x

        def _body_fun(i, val):
            # primal is the first order evaluation
            # tangent is the second order
            primal, tangent = jax.jvp(grad_f_closure, (walkers,), (eye[..., i],))
            return val + primal[i]**2 + tangent[i]

        # from lower to upper
        # (lower, upper, func(int, a) -> a, init_val)
        # this is like functools.reduce()
        # val is the previous  val (initialised to 0.0)
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


cfg = setup(system='LiSolid',
               #pretrain=True,
               n_pre_it=501,
               n_walkers=128,
               n_layers=2,
               n_sh=32,
               n_ph=8,
               opt='kfac',
               n_det=4,
               print_every=1,
               save_every=5000,
               lr=1e-3,
               n_it=10000,
               norm_constraint=1e-4,
               damping=1e-3,
               name='mic_w_interaction',
               kappa = 1.,
               real_cut = 6.,
               reciprocal_cut = 13)

key = rnd.PRNGKey(cfg['seed'])

mol = SystemAnsatz(**cfg)

wf, vwf, kfac_wf, wf_orbitals = create_wf(mol)
params = initialise_params(key, mol)
d0s = initialise_d0s(mol)

sampler, equilibrate = create_sampler(wf, vwf, mol, **cfg)

walkers = mol.initialise_walkers(**cfg)

print(walkers.shape)

